#=
# Using DPPs for recommender systems
The goal of this notebook is to build a simple recommender system using DPPs. We will construct a system that recommends a random selection of jokes, but that ensures that the jokes are sufficiently diverse (you won't have to read three jokes where a horse walks into a bar). In addition, the system will try to ensure that the jokes are funny (quality). We will see that there is a quality-diversity trade-off: if we emphasise higher-quality jokes, we have a smaller set to choose from.
=#

#=
We use an old dataset called the Jester dataset (Goldberg et al., 2001), available online at https://goldberg.berkeley.edu/jester-data/. It contains a database of 100 jokes, rated by anonymous users. The data was collected between 1999 and 2003. Content warning: some of these jokes are dated and off-colour. We will use the user ratings to score jokes by how funny the users found them.
=#

##Load some packages
using Plots, Statistics, LinearAlgebra, JLD2

##Load the dataset
@load "recommendation/jester_data.jld2" jester_ratings jester_jokes;

#=

## Measuring quality

Each row in jester_ratings contains the ratings of a single user. Jokes are rated from -10 to 10, missing ratings are encoded as missing values. Some users are more enthusiastic than others (they give higher scores on average). We will need to take this into account when computing the average quality of a joke. Other users always give the same score, so we filter them out.
=#

user_mean = map(mean ∘ skipmissing, eachrow(jester_ratings));
user_sd = map(std ∘ skipmissing, eachrow(jester_ratings));
histogram(user_mean, title = "Mean user rating")
scatter(user_mean, user_sd, xlab = "Mean rating", ylab = "Std. dev of rating")

keep = user_sd .> 0.5
valid_ratings = jester_ratings[keep, :];

##Now we standardise the ratings
scaled_ratings = (valid_ratings .- user_mean[keep]) ./ user_sd[keep];

# Now that the scores are cleaned up and standardised, we can compute an aggregate score for each joke. Here we use the median rating.

joke_scores = map(median ∘ skipmissing, eachcol(scaled_ratings));
joke_sds = map(std ∘ skipmissing, eachcol(scaled_ratings));
## Higher-rated jokes are more consensual across users
scatter(joke_scores, joke_sds, xlab = "Median joke score", ylab = "Std. dev of score")

## Show the "best" and "worst" jokes
sp=sortperm(joke_scores)
println("Worst-rated joke: \n " * jester_jokes[sp[1]] * "\n")
println("Best-rated joke: \n " * jester_jokes[sp[end]] * "\n")


#=

## Similarity via text embeddings

We have the first ingredient in our recommender system, a quality score.
We now need a way to measure how similar two jokes are. In the old-days you'd use hand-made features like TF-IDF, nowadays we can just use a transformer.
In this case we ran the jokes through [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), using the Python SentenceTransformer package. This model is designed to produce "embeddings" for short text fragments.
An embedding represents a text fragment as a vector of fixed dimension (here, it's a vector in $\mathbb{R}^384$)
The model is trained on a large set of examples, consisting of similar and dissimilar fragments. Suppose $t$,$t_+$,and $t_{-}$ are, respectively, a piece of text to be matched, a positive match, and a non-match.
Then the model $g$ is trained such that its output $g(t)$ is more similar to $g(t_{+})$ then $g(t_{-})$, ie $g(t)^\top g(t_+) \geq g(t)^\top g(t_{-})$.
We are going to exploit this by using the model output as a vector representation of each joke (a feature vector)
=#

@load "recommendation/jester_data.jld2" jester_embeddings;

#=
The matrix of embeddings is a 100x384 matrix, where each row is a vector of features corresponding to a joke. Let's see if the embeddings are sensible. We're going to find the most similar jokes to joke 16, according to our embeddings
=#

println("Target joke: "*jester_jokes[16]*"\n")
## Compute similarity:
sim=jester_embeddings*jester_embeddings[16, :];
sp=sortperm(sim, rev = true)
for i = 1:4
    println("Most similar $i: "*jester_jokes[sp[i+1]]*"\n")
end

#=
The first three are puns in a Q&A format, then the next one is a pun.
The target joke is a pun in Q&A format, so that's pretty good!

## Similarity and quality

In order to define a DPP, or, in this case, an L-ensemble, we need a measure of pairwise similarity between our items. Here, we use the embeddings:

=#


S = jester_embeddings*jester_embeddings';

#=
If we were to use S alone, we'd just be getting a random selection of diverse jokes. In addition to similarity, we use a quality/similarity decomposition, as in Kulesza & Taskar (2011). Quality is represented by a diagonal matrix.
=#

Q = Diagonal(exp.(joke_scores));

#=
The L-ensemble will be defined by the matrix Q*S*Q'.
=#
using Determinantal
ell = EllEnsemble(Q*S*Q');

#=
In order to get a certain number of jokes on average (here 5), we need to rescale the L-ensemble
=#
rescale!(ell, 5)
#=
What this does is rescale the L-ensemble to $αQSQ^\top$
with α, a scalar, set such that the expected sample size is 5.
=#

##Here's a sample of jokes
Determinantal.sample(ell);
#=
Notice that running the sample command several times gives sets of different sizes! Standard DPPs have variable size. Here the mean size is 5, but there's variance. This can be examined by calling size_statistics
=#
size_statistics(ell);

#=
The mean is at 5 as expected, but the standard deviation is ~2. That gives a fair chance of returning no jokes at all. If we want a sample with a fixed size, one way is to use a fixed-size DPP (AKA k-DPP).
=#

Determinantal.sample(ell, 5); ##returns a set of size 5, always

#=
Another way is to modify the rank of the DPP, or play with the eigenvalues - this is a more advanced topic.
=#


#=
For now, let's have a look at the inclusion probabilities.
The inclusion probability of joke $i$ is the prob. that it appears
in the random set produced by the DPP.
=#

scatter(joke_scores, inclusion_prob(ell), xlab = "Score", ylab = "Prob. inclusion") ##Standard DPP
scatter!(joke_scores, inclusion_prob(ell, 5)) ##Fixed-size DPP (nearly equal)

#=
Notice how the inclusion probabilities mostly scale with the quality index,
but not perfectly. If your goal is to have inclusion probabilities that are a
strict function of the quality index, use match_prob. It will produce a different diagonal scaling $Q$, set such that inclusion probabilities are as specified (if possible).
=#

q = exp.(joke_scores); ##We want our inclusion probabilities to be proportional to q
q = (q/sum(q))*5; ##We want them to sum to 5, because that's the target size
ell_matched = match_prob(EllEnsemble(S), q); ##This calls a numerical minimisation routine, it may be a bit slow
scatter(joke_scores, inclusion_prob(ell_matched), xlab = "Score", ylab = "Prob. inclusion")
