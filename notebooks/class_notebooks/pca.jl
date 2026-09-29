### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 2ee6479b-9077-4618-9496-2a954a38ca06
begin
	if "PROJECT_DIR" ∈ keys(ENV)
		using Pkg; Pkg.activate(ENV["PROJECT_DIR"])
	end
	using ColorSchemes, Images
	using PlutoUI
	using Plots, Measures
	using LaTeXStrings
	using Statistics, LinearAlgebra
	using Random
	using RDatasets
end

# ╔═╡ 6e4e429c-1a3e-11ec-2d15-fbd7c7918109
md"""
# Principal component analysis

#### Learning Objectives

After completing this worksheet, you should be able to:

* Define and explain the rank of a matrix.
* Understand how Principal Component Analysis (PCA) works and its applications.
* Apply PCA to a dataset to reduce noise and identify patterns.





[Principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) can be thought of as an application or way of interpreting the SVD.


This method tries to answer the questions "which 'directions' are the most important  in the data" and "can we [reduce the dimensionality](https://en.wikipedia.org/wiki/Dimensionality_reduction) (number of useful variables) of the data"?

This can be viewed as a method for finding and exploiting **structure** in data. It also leads towards ideas of **machine learning**.
"""

# ╔═╡ d6e34aa0-b118-493e-93cb-401fe4cfbc90
PlutoUI.TableOfContents()

# ╔═╡ 0b3b3c55-a583-4eca-a5e3-8614db4d5df3
md"""
# Rank of a matrix
"""

# ╔═╡ 5ad06b8d-dce7-4e91-a58c-96f51ab1cd0b
md"## Flags"

# ╔═╡ 4ce44f19-c355-49bf-83e7-f3cbb683c26a
md"""Given two vectors of size $m$ × 1 and $n$ × 1 respectively

$$\mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \\ \vdots \\ u_m \end{bmatrix},
\quad
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

their [outer product](https://en.wikipedia.org/wiki/Outer_product), $u ⊗ v$ is defined as the $m × n$ matrix $A$ obtained by multiplying each element of $u$ by each element of $v$:

$$
  \mathbf{u} \otimes \mathbf{v} = \mathbf{A} =
  \begin{bmatrix}
    u_1v_1 & u_1v_2 &  \dots & u_1v_n \\
    u_2v_1 & u_2v_2 &  \dots & u_2v_n \\
    \vdots & \vdots & \ddots & \vdots \\
    u_mv_1 & u_mv_2 &  \dots & u_mv_n
  \end{bmatrix}$$

"""

# ╔═╡ 5f93b743-19f1-465b-a1a5-0f933e2b6978
outer(v, w) = [x * y for x in v, y in w]

# ╔═╡ fed11713-0e44-41da-bb58-93f5efa15f33
outer(1:10, 1:12)

# ╔═╡ 53817eb4-9a88-4e8f-b7d3-55995433a041
md"""**...or:**"""

# ╔═╡ 72b34791-8cfe-4b52-aeb2-464651bd9a6d
collect(1:10) * collect(1:12)'

# ╔═╡ 4c6274f7-2c82-43ff-9098-daf9b7e73c38
md"""
Each column is a multiple (in general not an integer multiple) of every other column; and each row is a multiple of every other row.

This is an example of a **structured matrix**, i.e. one in which we need to store *less information* to store the matrix than the full $(m \times n)$ table of numbers (even though no element is 0). For example, for an outer product we only need $m + n$ numbers, which is usually *much* less.
"""

# ╔═╡ 0c801b36-bb69-49d4-8e69-47b9c50d5c93
flag = outer([1, 0.1, 2], ones(6))

# ╔═╡ 31f80508-f952-4c08-b214-fc0aaee17456
ones(6)

# ╔═╡ 59588c80-2c1e-4058-a940-b3d8fec4fc8b
flag2 = outer([1, 0.1, 2], [1, 1, 1, 3, 3, 3])

# ╔═╡ 745744db-15b3-4d93-be8d-40fab6f0976d
md"""
Note that outer products are not always immediate to recognise just by looking at an image! But you should be able to recognise that there is some kind of structure.
"""

# ╔═╡ 77d22611-7428-46cb-9c7c-a9f5741b2c90
md"## Matrix rank"

# ╔═╡ ca27c917-0ac0-4a3b-a4e7-9837ac92f4ba
md"""
If a matrix can be written exactly as a *single* multiplication table / outer product, we say that its **rank** is 1, and we call it a **rank-1** matrix. Similarly, if it can be written as the *sum* of *two* outer products, it has **rank 2**, etc.
"""

# ╔═╡ b91893f7-8432-451f-8da3-f5bc6a0dbe3f
md"Let's see what a random rank-1 matrix looks like:"

# ╔═╡ bc7ff070-5e58-4c17-aa4b-11f05f895811
w = 75

# ╔═╡ f5a08b12-1300-44a8-86cd-09aa274fcd20
begin
	Random.seed!(1) # Set the random seed so we get the same image every time (which makes grading easier).
	image = outer([1; 0.4; rand(50)], rand(w));
end


# ╔═╡ 3ac6bbab-ab6c-4721-9354-134227784830
md"""
It has a characteristic checkerboard or patchwork look.
"""

# ╔═╡ fdd46e40-b46e-494b-8329-a2e25e6c5b98
md"""
Here's a random rank-2 matrix:
"""

# ╔═╡ f70c2663-0a47-43b2-ac90-c915e0f7aeef
md"""
We see that it starts to look less regular.
"""

# ╔═╡ e5512127-44d6-4c02-8111-085e1c404c0a
md"""
## Effect of noise
"""

# ╔═╡ a4ac5416-ce06-4075-b6db-c06ff4aca534
md"""
Now what happens if we add a bit of **noise**, i.e. randomness, to a rank-1 matrix?
"""

# ╔═╡ 13c4c67a-d7ed-4094-9aee-062ab5a71ce7
begin
	Random.seed!(1) # Set the random seed so we get the same image every time (which makes grading easier).
	noisy_image = image .+ 0.03 .* randn.();
end;

# ╔═╡ 1fa453a6-41e1-476f-8807-7b5ba8d7d13c
md"""The noisy matrix now has a rank larger than 1. But visually we can see that it is "close to" the original rank-1 matrix.

Given this matrix, how could we discover that it is close to a structured, rank-1 matrix? We would like to be able to find this out and say that the matrix is close to a simple one.

> 👉 What happens if you change the amount of noise?

"""

# ╔═╡ dfaf70a6-7d70-4708-8c6e-d5e2cf26d3d4
md"## Images as data"

# ╔═╡ e9d63d6a-9304-4706-919f-ac332ac21fbc
md"""

Now let's treat the image as a **data matrix**, so that each column of the image / matrix is a **vector** representing one observation of data. (In data science it is often the rows that correspond to observations.)

Let's try to visualize those vectors, taking just the first two rows of the image as the $x$ and $y$ coordinates of our data points:
"""

# ╔═╡ 687ad22d-649d-4333-b82f-0724eb313ce2
image[1:2, 1:20]

# ╔═╡ b5b89b95-ac89-4575-8067-3f451619af35
begin
	xx = image[1, :]
	yy = image[2, :]
end;

# ╔═╡ 95418bff-b409-485c-8b61-715605304c5c
md"# From images to data"

# ╔═╡ b841f27a-2487-4bff-ab34-aa4f33aeac2b
md"""
We would like to **visualise** this data with the `Plots.jl` package, passing it the $x$ and $y$ coordinates as arguments and plotting a point at each $(x_i, y_i)$ pair.
We obtain the following plot of the original rank-1 matrix and the noisy version:
"""

# ╔═╡ 0a8f9bed-307b-4481-8576-7fd758232839
begin
	xs = noisy_image[1, :]
	ys = noisy_image[2, :]

	scatter(xs, ys, label="noisy",
		alpha=0.3, ms=4, ratio=1)

	scatter!(image[1,:], image[2,:], label="rank-1", alpha=0.3, m=:square,
		ms=4, framestyle=:origin)

	title!("Plotting 2 rows of a rank-1 matrix gives a straight line!")
end

# ╔═╡ b3acd434-7ceb-4487-b49e-d1fdfec901e6
md"We see that the exact rank-1 matrix has columns that **lie along a line** through the origin in this representation, since they are just multiples of one another.

E.g. If $(x_1, y_1)$ and $(x_2, y_2)$ are two columns with $x_2 = cx_1$ and $y_2 = cy_1$, then $y_2 / x_2 = y_1 / x_1$, so they lie along the same line through the origin.

The approximate rank-1 matrix has columns that **lie *close to* the line**!"

# ╔═╡ cc371907-1f37-4798-bd3a-8b5b83c72ff9
md"So, given the data, we want to look at it do see if it lies close to a line or not.
How can we do so in an *automatic* way?
"

# ╔═╡ 9f1e16ef-0933-412b-badf-c4b6db19325f
md"""
## Measuring data cloud "size" -- using statistics
"""

# ╔═╡ ce6fe51e-4b23-4899-a344-0312378ef26b
md"Looking at this cloud of data points, a natural thing to do is to try to *measure* it: How wide is it, and how tall?"

# ╔═╡ 0b0fde5d-01cb-4f64-9faf-162d8b41d0c6
md"""For example, let's think about calculating the width of the cloud, i.e. the range of possible $x$-values of the data. For this purpose the $y$-values are actually irrelevant.
"""

# ╔═╡ 36dff8e6-fa27-44d8-82d9-d7e128fe7933
md"""
A natural idea would be to just scan the data and take the maximum and minimum values. However, real data often contains anomalously large values called **outliers**, which would dramatically affect this calculation. Instead we need to use a **statistical** method where we weight the data and average over all the data points. This process will hopefully be affected less by outliers, and will give a more representative idea of the size of the **bulk** of the data.
"""

# ╔═╡ c0dc9067-bffd-4b2f-aaac-ecc3c42d599d
md"""
A first step in analysing data is often to **centre** the data around 0 by subtracting the mean, sometimes called "de-meaning":
"""

# ╔═╡ 42077f19-15c0-48da-963f-35795ddae9fe
begin
	xs_centered = xs .- mean(xs)
	ys_centered = ys .- mean(ys)
end

# ╔═╡ ea7de550-1015-43ac-8e1e-4b030b3873f7
scatter(xs_centered, ys_centered, ms=5, alpha=0.5, ratio=1, leg=false,
	framestyle=:origin)

# ╔═╡ 33072dc0-ae6e-4779-866c-3bbe85006e0a
md"""
## Measuring a "width" of a data set
"""

# ╔═╡ d0b94706-7719-4b37-a5ff-81fd75a2c97b
md"""
A natural way to measure the width of a data set could be to measure some kind of width *separately* in both the $x$ and $y$ directions, in other words by **projecting** the data onto one of the axes, while ignoring the other one. Let's start with the $x$ coordinates of the centred data and try to average them.

If we literally average them we will get $0$ (since we have subtracted the mean). We could ask "on average how far away from the origin is the data". This would give the [mean absolute deviation](https://en.wikipedia.org/wiki/Average_absolute_deviation):
"""

# ╔═╡ 9b6bb29c-7210-41b4-8714-d44f2795e247
begin
	scatter(xs_centered, ys_centered, ms=5, alpha=0.5,
		ratio=1, leg=false, framestyle=:origin)

	scatter!(xs_centered, zeros(size(xs_centered)), ms=5,
		alpha=0.1, ratio=1, leg=false, framestyle=:origin)

	for i in 1:length(xs_centered)
		plot!([(xs_centered[i], ys_centered[i]), (xs_centered[i], 0)],
			ls=:dash, c=:black, alpha=0.1)
	end

	plot!()

end

# ╔═╡ 433bc1de-13d7-440e-a27a-717e7666eece
mean(abs.(xs_centered))

# ╔═╡ e9a6b1b8-3933-426d-9d80-7e20274d92c9
md"""
This is a perfectly good computational measure of distance. However, there is another measure which is easier to reason about theoretically / analytically:
"""

# ╔═╡ 4573fa91-2544-4c4e-8bc2-009ccc4411ef
md"""
### Root-mean-square distance: Standard deviation
"""

# ╔═╡ 4193da38-9af3-473f-bf91-804b33581755
md"""
The **standard deviation** is the **root-mean-square** distance of the centered data from the origin.

In other words, we first *square* the distances (or displacements) from the origin, then take the mean of those, giving the **variance**. However, since we have squared the original distances, this gives a quantity with units "distance$^2$", so we need to take the square root to get back to a measurable length:
"""

# ╔═╡ f736bf30-0ce8-43cf-98a9-39011a106203
begin
	σ_x = √(mean(xs_centered.^2))   # root-mean-square distance from 0
	σ_y = √(mean(ys_centered.^2))
end

# ╔═╡ 625bd8c2-a76c-4bbc-a9ef-958d5336499f
md"This gives the following approximate extents (standard deviations) of the cloud:"

# ╔═╡ a4603da7-4da8-4cc1-9076-857363296629
begin
	scatter(xs_centered, ys_centered, ms=5, alpha=0.5, ratio=1, leg=false,
			framestyle=:origin)

	vline!([-2*σ_x, 2*σ_x], ls=:dash, lw=2, c=:green)
	hline!([-2*σ_y, 2*σ_y], ls=:dash, lw=2, c=:blue)

	annotate!( 2σ_x * 0.93, 0.03, text(L"2\sigma_x",  14, :green))
	annotate!(-2σ_x * 0.88, 0.03, text(L"-2\sigma_x", 14, :green))

	annotate!(0.05,  2σ_y * 1.13, text(L"2\sigma_y",  14, :blue))
	annotate!(0.06, -2σ_y * 1.14, text(L"-2\sigma_y", 14, :blue))

end

# ╔═╡ 0d21ea46-1ac5-40c7-a439-0c0e4d0538d9
md"""
We expect most (around 95%) of the data to be contained within the interval $\mu \pm 2 \sigma$, where $\mu$ is the mean and $\sigma$ is the standard deviation. (This assumes that the data is **normally distributed**, which is not actually the case for the data generated above.)
"""

# ╔═╡ f8fc96af-1e93-4a76-b6f3-66271cfd894c
md"## Correlated data"

# ╔═╡ b6318d2a-9b21-4b54-a5e9-e75d09867144
md"""
However, from the figure it is clear that $x$ and $y$ are not the "correct directions" to use for this data set. It would be more natural to think about other directions: the direction in which the data set is mainly pointing (roughly, the direction in which it's longest), together with the approximately perpendicular direction in which it is  narrowest.

We need to find *from the data* which directions these are, and the extent (width) of the data cloud in those directions.

However, we cannot obtain any information about those directions by looking *separately* at $x$-coordinates and $y$-coordinates, since within the same bounding boxes that we just calculated the data can be distributed in many different ways.

Rather, the information that we need is encoded in the *relationship* between the values of $x_i$ and $y_i$ *for the points in the data set*.

For our data set, when $x$ is large and negative, $y$ is also rather negative; when $x$ is 0, $y$ is near $0$, and when $x$ is large and positive, so is $y$. We say that $x$ and $y$ are **correlated** -- literally they are mutually ("co") related, such that knowing some information about one of them allows us to predict something about the other.

For example, if I measure a new data point from the same process and I find that $x$ is around $0.25$ then I would expect $y$ to be within the range $0.05$ to $0.2$, and it would be very surprising if $y$ were -0.5.

"""

# ╔═╡ 7f689126-4b0f-4642-a0c3-6530dcb18017
md"""
We want to think about different *directions*, so let's introduce an angle $\theta$ to describe the direction along which we are looking. We want to calculate the width of the cloud *along that direction*.

Effectively we are *changing coordinates* to a new coordinate, oriented along the line.
"""

# ╔═╡ 158b9087-88ca-4270-8181-f7f1a09f6f08
md"## Rotating the axes"

# ╔═╡ 6007ea5a-2a3b-4743-b057-524aedda1a12
md"""By rotating the axes we can "look in different directions" and calculate the width of the data set "along that direction". What we are really doing is a perpendicular **projection** of the data onto that direction."""

# ╔═╡ 7767ec92-cd06-438e-b20c-7c28a348ea43
begin
	M = [xs_centered ys_centered]';
	σs = svdvals(M);
	#variances = σs.^2 ./ 199

	imax = argmax(M[1, :]);

	R(θ)= [cos(θ) sin(θ)
		  -sin(θ) cos(θ)];

	variance(θ) = var( (R(θ) * M)[1, :] );
	variance(θ::AbstractArray) = variance(θ[1]);

	variances = variance.(range(0, 2π, length=361));
end;

# ╔═╡ 77f2b0de-b7fb-47df-8ffb-cbf628232548
md"""The direction in which the variance is **maximised** gives the most important direction: It is the direction along which the data "points", or the direction which best distinguishes different data points. This is often called the first **principal component** in statistics, or the first **singular vector** in linear algebra.

"""

# ╔═╡ f5f80549-5e39-4bcb-9d16-3323b31bba6d
md"""
## Exercise 1

Choose a coordinate system rotation θ that maximizes variance in direction θ.
"""

# ╔═╡ 4157f23c-d591-4513-8fd1-f75d11240529
md"""
θ (degrees) = $(@bind degrees Slider(0:360, default=0, show_value=true))
"""

# ╔═╡ 3761875e-2cc8-47a2-a891-0622feddc7f2
begin
	θ = π * degrees / 180   # radians

	p1 = begin
		scatter(M[1, :], M[2, :], ratio=1, leg=false, ms=2.5, alpha=0.5,
				framestyle=:origin, margin=0mm)

		projected = ([cos(θ) sin(θ)] * M) .* [cos(θ) sin(θ)]'
		scatter!(projected[1, :], projected[2, :], m=3, alpha=0.1, c=:green)


		lines_x = reduce(vcat, [M[1, i], projected[1, i], NaN] for i in 1:size(M, 2))
		lines_y = reduce(vcat, [M[2, i], projected[2, i], NaN] for i in 1:size(M, 2))

		plot!(lines_x, lines_y, ls=:dash, c=:black, alpha=0.1)

		plot!([0.7 .* (-cos(θ), -sin(θ)), 0.7 .* (cos(θ), sin(θ))], lw=1,
			arrow=true, c=:red, alpha=0.3)
		xlims!(-0.7, 0.7)
		ylims!(-0.7, 0.7)

		scatter!([M[1, imax]], [M[2, imax]],  ms=3, alpha=1, c=:yellow)

		title!("align arrow with cloud")
	end;

	p2 = begin

		M2 = R(θ) * M

		scatter(M2[1, :], M2[2, :],ratio=1, leg=false, ms=2.5, alpha=0.3,
			framestyle=:origin, size=(500, 500), margin=0mm)

		xlims!(-0.7, 0.7)
		ylims!(-0.7, 0.7)

		scatter!(M2[1, :], zeros(size(xs_centered)), ms=3, alpha=0.1, ratio=1,
			leg=false, framestyle=:origin, c=:green)


		lines2_x = reduce(vcat, [M2[1, i], M2[1, i], NaN] for i in 1:size(M2, 2))
		lines2_y = reduce(vcat, [M2[2, i], 0, NaN] for i in 1:size(M2, 2))

		plot!(lines2_x, lines2_y, ls=:dash, c=:black, alpha=0.1)

		σ = std(M2[1, :])
		vline!([-2σ, 2σ], ls=:dash, lw=2)

		title!("σ = $(round(σ, digits=4))")

		annotate!(2σ+0.05, 0.05, text("2σ", 10, :green))
		annotate!(-2σ-0.05, 0.05, text("-2σ", 10, :green))
	end

	#p3 = begin
	#	plot(0:360, variances, leg=false)
	#	scatter!([degrees], [σ^2])
	#	xlabel!("θ")
	#	ylabel!("variance in ➡ θ")
	#end

	#l = @layout[
	#	grid(1,2)
	#	a{0.2h}
	#]

	# p3, layout=l,
	plot(p1, p2, size=(800, 500))
end

# ╔═╡ e5bfc281-9990-4ae5-b60b-aaed1e77f94c
md"""
Congratulations! You've just found the first principal component the hard way! You could subtract the first principal component from the data and then repeat the process to find the second principal component, and so on.

Luckily, we won't have to do that because, as we learned in the pre-lecture video, we can find the principal components automatically from the SVD of our matrix.

But first...

#### 🤔 Another way to think about it!

If you remember from the last class, the SVD says that:

```math
U \times \sigma \times V^t = X.
```

A different way to think about the exercise above is that the first column of `U` is represented the cosine and the sine of the angle that we're rotating the matrix by:
"""

# ╔═╡ 4ecdab7b-280b-40da-a9d7-53c38d60c36d
U = [cosd(degrees), sind(degrees)]

# ╔═╡ 1b29ba54-2e32-4386-9be7-e64367cb1780
md"""
... and the first row of our rotated data on the righthand plot above is equal to `σ × Vt`:
"""

# ╔═╡ 2ee8d49e-85f1-43cc-9bd8-11f3a1dc9fba
sigma_times_Vt = M2[1:1, :]

# ╔═╡ 4967ccdd-7b05-47ab-af78-6f9c5722b1d2
md"""
This means that we can reconstruct an approximation of our original data by doing `U * sigma_times_Vt`:
"""

# ╔═╡ ef5b00a1-b3fd-43ea-a44a-c9a7c827c3e4
begin
	recon = U * sigma_times_Vt
	scatter(M[1,:], M[2, :], label="original data")
	scatter!(recon[1,:], recon[2, :], label="reconstructed data")
end

# ╔═╡ d15d8342-ea88-48d6-8cb7-965eafddc3ea
md"""
## Exercise 2

Now, we're going to do PCA the automatic way using SVD, rather than by hand.

Remember, for data matrix $$X$$:

```math
\begin{aligned}

B &= X - \bar{X} \\

T V^T &= B \\

T &= U \Sigma \\

\end{aligned}
```
where $T$ is a matrix of principal components, $V$ is a matrix of loadings, and $^T$ means transpose. Remember that you can find $U$, $\Sigma$, and $V^T$ by computing the SVD of $B$.

Let's try doing PCA on our full `52×75` `noisy_image` from above.
"""

# ╔═╡ 74bedf81-9814-44db-aa5b-229ebee45e7f
pca_data = noisy_image

# ╔═╡ 16c39f9f-08c5-44df-91a4-f6565c0e2145
md"""

Remember from above that noisy_image is approximately a rank-1 matrix, so we should be able to represent it well with just 2 PCA modes. (Each "PCA mode" corresponds to a SVD singular value.) In other words, we created our matrix by multiplying two vectors, and we're using PCA to figure out what those two vectors were.


Your assignment is to:

1. Calculate the SVD of the `pca_data` above and use that to calculate `T̃` and `Ṽ`, each of which should only include 2 PCA modes. (`T̃` and `Ṽ` are the same as $T$ and $V$ in the equation above, except `T̃` and `Ṽ` should only represent two PCA modes rather than the full matrix decomposition.)

2. Determine how much of the variance of the original data is represented within the first two PCA modes.

3. Reconstruct an approximation of the original matrix from the compressed representation. How similar do they look?

"""

# ╔═╡ 2dd83539-eb46-408a-a398-90303aeb1a94
md"""### Exercise 2.1

Calculate `T̃` and `Ṽt`.
"""

# ╔═╡ 177d86ac-ce25-45c9-930c-95539f4778b8
begin
	n_modes = 2
	X = pca_data
	X̄ = mean(X, dims=2)
	B = X .- X̄
	F = svd(B)
	T̃ = F.U[:, 1:n_modes] * Diagonal(F.S[1:n_modes])
	Ṽt = F.Vt[1:n_modes,:]
end

# ╔═╡ f0d97466-3522-40b6-9b41-8a840dd4846c
md"""
### Exercise 2.2

Calculate the fraction of the overall variance in the data captured by the first 2 PCA modes.
"""

# ╔═╡ f6e77219-62c5-4e58-a197-7395095a664d
frac_var = sum(F.S[1:n_modes].^2) / sum(F.S.^2)

# ╔═╡ 136a316b-bab1-454d-bcfa-3854681f8b89
md"""
### Exercise 2.3

Reconstruct an approximation of the original matrix from the compressed representation. How similar do they look?
"""

# ╔═╡ 0cec673a-5eb2-4554-b5a3-e2719bd977e1
reconstructed_data = T̃ * Ṽt .+ X̄

# ╔═╡ 7bf7aaa7-c005-4c61-b45b-33260147e8e1
md"""
# Using PCA for Data Science
"""

# ╔═╡ 31270f76-ec06-4c15-a60e-497d5017c0dd
md"## The Iris dataset"

# ╔═╡ 72a1c126-6575-4375-adec-b1bc77d9854f
md"""

Now we're going to be working again with a dataset that contains information about 150 specimens of irises. The dependent variable is a label of which species of iris each specimen is:

![irises](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)

The indpendent variables are the length and width of the petal and sepal of each specimen.

We want to be able to identify the type of an iris based on the measurements of its sepal and petal (or to identify the dpendent variable based on the indpendent variables).

The dataset is below:

"""

# ╔═╡ dc83e29d-680a-49c2-91db-ab7af33661bb
iris = dataset("datasets", "iris")

# ╔═╡ e600a641-9427-45ca-8e0c-6d2ccbeaa822
md"""## Classifying the types
When we have a dataset where we want to be able classify the different items in a dataset, one way to do it is by using simple cutoff values.
For example, in a classroom, students with > 90% get an A, students with 80–90% get a B, and so on.

We can do this with our Iris dataset, so see if we can classify the iris types based on different values of their sepal lengths:
"""

# ╔═╡ 3e2364ec-540f-4be3-bf9f-23369534a98b
begin
	histogram(iris.SepalLength[iris.Species .== "setosa"], bins=50, alpha=0.6,
		label="Setosa")
	histogram!(iris.SepalLength[iris.Species .== "versicolor"], bins=50, alpha=0.6,
		label="Versicolor")
	histogram!(iris.SepalLength[iris.Species .== "virginica"], bins=50, alpha=0.6,
		label="Virginica")
end

# ╔═╡ 99946c17-c84c-44ca-bbd4-6075997a0d17
md"""
As you can see, this plan doesn't work very well in this case. We can tell than Virginica tends to have a longer sepal length than Setosa, but Versicolor is right in the middle and there's no way to divide the types in a way that doesn't mis-classify a lot of the dataset.

Maybe PCA could help us with this task?
"""

# ╔═╡ 2c16bb6c-d3af-47aa-8563-916cafbb2c49
md"""### Exercise 3: Calculate the PCA of the Iris dataset

The situation here is a little different than above, because our data is in a DataFrame rather than a matrix.
However we can convert a DataFrame `df` to a matrix like:

```julia
m = Matrix(df)
```

If we just want to convert the first `n` columns of the DataFrame to a matrix, then it would be:

```julia
m = Matrix(df[:, 1:n])
```

👉 Your task is to calculate `T̃` and `Ṽt` for this dataset.
To do this, you can use the same steps as in exercise 2, keeping in mind that:

1. You only want to use the attributes of the irises to do the PCA, i.e. the first four columns of the dataset but `Not(:Species)`.
2. In the DataFrame, each row is an observation (i.e. a flower), but for PCA we want each column to be an observation.
"""

# ╔═╡ 94895712-5ae0-49e7-8960-d081a6e92268
begin
	n_modes_iris = 1
	X_iris = Matrix(iris[:, 1:4])'
	X̄_iris = mean(X_iris, dims=2)
	B_iris = X_iris .- X̄_iris
	F_iris = svd(B_iris) # F should be the result of your SVD calculation.
	T̃_iris = F_iris.U[:, 1:n_modes_iris] * Diagonal(F_iris.S[1:n_modes_iris])
	Ṽt_iris = F_iris.Vt[1:n_modes_iris,:]
end

# ╔═╡ 475384f5-4d81-4676-9878-b0489eca9781
md"""
Now, if we make the same plot with the first PCA mode rather than one of the measured attributes, we can see that it works much better, allowing us to say that Virginica irises are those where the value is < ~-0.045, Versicolors are the remaining ones where the value is < ~0.05, and Setosas are the remainder after that.

The reason that this works is because what SVD/PCA does is to "rotate" the data or put it into a different coordinate system where the first dimension of that coordinate system explains as much of the variability in the dataset as possible, thus increasing the chance that we will be able to differentiate among the different classes.
"""

# ╔═╡ ddb2ee4c-3d11-48b1-9487-937a3efccfe5
ismissing(Ṽt_iris) ? missing : begin
	histogram(Ṽt_iris[1, iris.Species .== "setosa"], bins=50, alpha=0.6,
		label="Setosa")
	histogram!(Ṽt_iris[1, iris.Species .== "versicolor"], bins=50,  alpha=0.6,
		label="Versicolor")
	histogram!(Ṽt_iris[1, iris.Species .== "virginica"], bins=50, alpha=0.6,
		label="Virginica")
	plot!([-0.045, -0.045], [0, 7], lc=:black, lw=3, label="Setosa<>Versicolor")
	plot!([0.05, 0.05], [0, 7], lc=:black, lw=3, label="Versicolor<>Virginica")
end

# ╔═╡ e04d2322-68e4-4355-9f1b-cb47d529769c
md"""

🎉 That's it!

Utility functions are below.

---

---

---

"""

# ╔═╡ 6b4b4891-bfd9-4aa0-a8cd-b7662ffc2f8b
begin
	almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]))

	still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]))

	keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]))

	yays = [md"Fantastic!", md"Splendid!", md"Great!", md"Yay ❤", md"Great! 🎉", md"Well done!", md"Keep it up!", md"Good job!", md"Awesome!", md"You got the right answer!", md"Let's move on to the next section."]

	correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]))

	hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))

	promising(text) = Markdown.MD(Markdown.Admonition("correct", "Looks promising!", [text]))

	not_defined(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define a variable called **$(Markdown.Code(string(variable_name)))**"]))

	show_image(M) = get.(Ref(ColorSchemes.rainbow), M ./ maximum(M))
	show_image(x::AbstractVector) = show_image(x')

end

# ╔═╡ 9810378f-0298-4637-b3bf-0038f9278109
show_image(flag)

# ╔═╡ eb7e304e-bffc-4b55-878f-6bb926a6fc81
show_image(flag2)

# ╔═╡ 07c0ed72-179f-4d92-9f57-d8d65fa4ffee
show_image(image)

# ╔═╡ 99af8c16-5001-43e1-ad35-f6b9a42bf563
begin
	image2 = outer([1; 0.4; rand(50)], rand(w)) +
	         outer(rand(52), rand(w))

	show_image(image2)
end

# ╔═╡ e1943831-4e2f-4de7-8dd2-b31a9b78beec
show_image(image)

# ╔═╡ 0cdd6877-b7dd-46ae-a4bd-f4da2fee891d
show_image(noisy_image)

# ╔═╡ 8cbf06a1-c090-4564-a2e0-bce05ce9a9b9
show_image(image[1:2, 1:20])

# ╔═╡ 7399d827-4cd7-49b5-9b7e-8c5cff050bd8
let
	if σ^2 > 0.997 * maximum(variances)
		correct()
	else
		keep_working()
	end
end

# ╔═╡ c7a111e8-4899-4f81-a10f-64ca9a0eed72
if !@isdefined(X)
	not_defined(:X)
else
	let
		a = X
		if ismissing(a)
			still_missing()
		elseif !(X ≈ pca_data)
			keep_working("`X` is not quite right")
		else
			correct(md"You've got `X` correct.")
		end
	end
end

# ╔═╡ a7d00915-e380-42b0-8a1a-bc653ff8010f
if !@isdefined(B)
	not_defined(:B)
else
	let
		a = B
		if ismissing(a)
			still_missing()
		elseif !(abs(mean(B)) < 1.0e-10)
			keep_working(md"`B` should have a mean of 0")
		elseif size(a) != (52, 75)
			keep_working(md"The dimensions of `B` should be (52,75)")
		elseif norm(B) ≉ 10.333055251396367
			keep_working(md"B is not quite right.")
		else
			correct(md"You've got `B` correct.")
		end
	end
end

# ╔═╡ d68285d6-719d-4d84-8a1b-6149c6acd8ea
if !@isdefined(T̃)
	not_defined(:T̃)
else
	let
		a = T̃
		if ismissing(a)
			still_missing()
		elseif size(a) != (52, 2)
			keep_working(md"The dimensions of `T̃` should be (52,2)")
		elseif norm(a,1) ≉ 64.87407523700043
			keep_working(md"`T̃` is not quite right.")
		else
			correct(md"You've got `T̃` correct.")
		end
	end
end

# ╔═╡ 7954abc9-b9ed-4240-a7fb-26738eae9807
if !@isdefined(Ṽt)
	not_defined(:Ṽt)
else
	let
		a = Ṽt
		if ismissing(a)
			still_missing()
		elseif size(a) != (2, 75)
			keep_working(md"The dimensions of `Ṽt` should be (2,75)")
		elseif norm(Ṽt,1) ≉ 15.105637568375604
			keep_working(md"`Ṽt` is not quite correct.")
		else
			correct(md"You've got `Ṽt` correct.")
		end
	end
end

# ╔═╡ 5669704a-57ae-4db5-a37b-252f83ddd8a7
if !@isdefined(frac_var)
	not_defined(:frac_var)
else
	let
		a = frac_var
		if ismissing(a)
			still_missing()
		elseif !isa(a, Number)
			keep_working()
		else
			if a ≉ 0.9697871629782244
				keep_working()
			else
				correct()
			end
		end
	end
end

# ╔═╡ 42f22aa5-df48-4d30-9185-df6efede8191
if !@isdefined(reconstructed_data)
	not_defined(:reconstructed_data)
else
	let
		a = reconstructed_data
		if ismissing(a)
			still_missing()
		elseif !isa(a, Matrix)
			keep_working(md"`transformed_data` should be a Matrix")
		else
			if size(a) != (52,75)
				keep_working(md"Your matrix is not the correct size")
			elseif norm(abs.(reconstructed_data-pca_data)) / length(pca_data) > 0.001
				keep_working()
			else
				correct()
			end
		end
	end
end

# ╔═╡ 25cb2a5d-6d2d-41a9-b7c6-bfa4e616c0f0
if !ismissing(reconstructed_data)
	plot(plot(show_image(pca_data), title="Original"),
		plot(show_image(reconstructed_data), title="Reconstructed"), size=(800, 300))
end

# ╔═╡ fda1783b-e760-4e70-9d1c-0665674548f7
if !@isdefined(X_iris)
	not_defined(:X_iris)
else
	let
		a = X_iris
		if ismissing(a)
			still_missing()
		elseif !(a isa AbstractMatrix)
			keep_working(md"`X_iris` should be a matrix.")
		elseif size(a) != (4,150)
			keep_working(md"The size of `X_iris` is not quite right. Do you have one observation in each column?")
		elseif !(norm(a) ≈ 97.66928892952994)
			keep_working(md"`X_iris` is not quite right.")
		else
			correct(md"You've got `X_iris` correct.")
		end
	end
end

# ╔═╡ b9b68558-e18e-47ce-843b-a71431738495
if !@isdefined(B_iris)
	not_defined(:B_iris)
else
	let
		a = B_iris
		if ismissing(a)
			still_missing()
		elseif !(a isa AbstractMatrix)
			keep_working(md"`B_iris` should be a matrix.")
		elseif size(a) != (4,150)
			keep_working(md"The size of `B_iris` is not quite right. Do you have one observation in each column?")
		elseif !(norm(a) ≈ 26.103076447039726)
			keep_working(md"`B_iris` is not quite right.")
		else
			correct(md"You've got `B_iris` correct.")
		end
	end
end

# ╔═╡ 71a82c6f-93fd-4c62-acca-898584960034
if !@isdefined(T̃_iris)
	not_defined(:T̃_iris)
else
	let
		a = T̃_iris
		if ismissing(a)
			still_missing()
		elseif !(a isa AbstractMatrix)
			keep_working(md"`T̃_iris` should be a matrix.")
		elseif size(a) != (4,1)
			keep_working(md"The size of `T̃_iris` is not quite right. Do you have one observation in each column?")
		elseif !(norm(a) ≈ 25.09996044218387)
			keep_working(md"`T̃_iris` is not quite right.")
		else
			correct(md"You've got `T̃_iris` correct.")
		end
	end
end

# ╔═╡ f9f17fc1-3540-4dec-8fb4-939bf6e35310
if !@isdefined(Ṽt_iris)
	not_defined(:Ṽt_iris)
else
	let
		a = Ṽt_iris
		if ismissing(a)
			still_missing()
		elseif !(a isa AbstractMatrix)
			keep_working(md"`Ṽt_iris` should be a matrix.")
		elseif size(a) != (1,150)
			keep_working(md"The size of `Ṽt_iris` is not quite right.")
		elseif !(norm(a, 3) ≈ 0.4665567953771965)
			keep_working(md"`Ṽt_iris` is not quite right.")
		else
			correct(md"You've got `Ṽt_iris` correct.")
		end
	end
end

# ╔═╡ Cell order:
# ╟─6e4e429c-1a3e-11ec-2d15-fbd7c7918109
# ╠═2ee6479b-9077-4618-9496-2a954a38ca06
# ╟─d6e34aa0-b118-493e-93cb-401fe4cfbc90
# ╟─0b3b3c55-a583-4eca-a5e3-8614db4d5df3
# ╟─5ad06b8d-dce7-4e91-a58c-96f51ab1cd0b
# ╟─4ce44f19-c355-49bf-83e7-f3cbb683c26a
# ╠═5f93b743-19f1-465b-a1a5-0f933e2b6978
# ╠═fed11713-0e44-41da-bb58-93f5efa15f33
# ╟─53817eb4-9a88-4e8f-b7d3-55995433a041
# ╠═72b34791-8cfe-4b52-aeb2-464651bd9a6d
# ╟─4c6274f7-2c82-43ff-9098-daf9b7e73c38
# ╠═0c801b36-bb69-49d4-8e69-47b9c50d5c93
# ╠═31f80508-f952-4c08-b214-fc0aaee17456
# ╠═9810378f-0298-4637-b3bf-0038f9278109
# ╠═59588c80-2c1e-4058-a940-b3d8fec4fc8b
# ╠═eb7e304e-bffc-4b55-878f-6bb926a6fc81
# ╟─745744db-15b3-4d93-be8d-40fab6f0976d
# ╟─77d22611-7428-46cb-9c7c-a9f5741b2c90
# ╟─ca27c917-0ac0-4a3b-a4e7-9837ac92f4ba
# ╟─b91893f7-8432-451f-8da3-f5bc6a0dbe3f
# ╠═bc7ff070-5e58-4c17-aa4b-11f05f895811
# ╠═f5a08b12-1300-44a8-86cd-09aa274fcd20
# ╠═07c0ed72-179f-4d92-9f57-d8d65fa4ffee
# ╟─3ac6bbab-ab6c-4721-9354-134227784830
# ╟─fdd46e40-b46e-494b-8329-a2e25e6c5b98
# ╠═99af8c16-5001-43e1-ad35-f6b9a42bf563
# ╟─f70c2663-0a47-43b2-ac90-c915e0f7aeef
# ╟─e5512127-44d6-4c02-8111-085e1c404c0a
# ╟─a4ac5416-ce06-4075-b6db-c06ff4aca534
# ╠═13c4c67a-d7ed-4094-9aee-062ab5a71ce7
# ╠═e1943831-4e2f-4de7-8dd2-b31a9b78beec
# ╠═0cdd6877-b7dd-46ae-a4bd-f4da2fee891d
# ╟─1fa453a6-41e1-476f-8807-7b5ba8d7d13c
# ╟─dfaf70a6-7d70-4708-8c6e-d5e2cf26d3d4
# ╟─e9d63d6a-9304-4706-919f-ac332ac21fbc
# ╠═8cbf06a1-c090-4564-a2e0-bce05ce9a9b9
# ╠═687ad22d-649d-4333-b82f-0724eb313ce2
# ╠═b5b89b95-ac89-4575-8067-3f451619af35
# ╟─95418bff-b409-485c-8b61-715605304c5c
# ╟─b841f27a-2487-4bff-ab34-aa4f33aeac2b
# ╠═0a8f9bed-307b-4481-8576-7fd758232839
# ╟─b3acd434-7ceb-4487-b49e-d1fdfec901e6
# ╟─cc371907-1f37-4798-bd3a-8b5b83c72ff9
# ╟─9f1e16ef-0933-412b-badf-c4b6db19325f
# ╟─ce6fe51e-4b23-4899-a344-0312378ef26b
# ╟─0b0fde5d-01cb-4f64-9faf-162d8b41d0c6
# ╟─36dff8e6-fa27-44d8-82d9-d7e128fe7933
# ╟─c0dc9067-bffd-4b2f-aaac-ecc3c42d599d
# ╠═42077f19-15c0-48da-963f-35795ddae9fe
# ╠═ea7de550-1015-43ac-8e1e-4b030b3873f7
# ╟─33072dc0-ae6e-4779-866c-3bbe85006e0a
# ╟─d0b94706-7719-4b37-a5ff-81fd75a2c97b
# ╟─9b6bb29c-7210-41b4-8714-d44f2795e247
# ╠═433bc1de-13d7-440e-a27a-717e7666eece
# ╟─e9a6b1b8-3933-426d-9d80-7e20274d92c9
# ╟─4573fa91-2544-4c4e-8bc2-009ccc4411ef
# ╟─4193da38-9af3-473f-bf91-804b33581755
# ╠═f736bf30-0ce8-43cf-98a9-39011a106203
# ╟─625bd8c2-a76c-4bbc-a9ef-958d5336499f
# ╟─a4603da7-4da8-4cc1-9076-857363296629
# ╟─0d21ea46-1ac5-40c7-a439-0c0e4d0538d9
# ╟─f8fc96af-1e93-4a76-b6f3-66271cfd894c
# ╟─b6318d2a-9b21-4b54-a5e9-e75d09867144
# ╟─7f689126-4b0f-4642-a0c3-6530dcb18017
# ╟─158b9087-88ca-4270-8181-f7f1a09f6f08
# ╟─6007ea5a-2a3b-4743-b057-524aedda1a12
# ╟─7767ec92-cd06-438e-b20c-7c28a348ea43
# ╟─3761875e-2cc8-47a2-a891-0622feddc7f2
# ╟─77f2b0de-b7fb-47df-8ffb-cbf628232548
# ╟─f5f80549-5e39-4bcb-9d16-3323b31bba6d
# ╟─4157f23c-d591-4513-8fd1-f75d11240529
# ╟─7399d827-4cd7-49b5-9b7e-8c5cff050bd8
# ╟─e5bfc281-9990-4ae5-b60b-aaed1e77f94c
# ╠═4ecdab7b-280b-40da-a9d7-53c38d60c36d
# ╟─1b29ba54-2e32-4386-9be7-e64367cb1780
# ╠═2ee8d49e-85f1-43cc-9bd8-11f3a1dc9fba
# ╟─4967ccdd-7b05-47ab-af78-6f9c5722b1d2
# ╠═ef5b00a1-b3fd-43ea-a44a-c9a7c827c3e4
# ╟─d15d8342-ea88-48d6-8cb7-965eafddc3ea
# ╠═74bedf81-9814-44db-aa5b-229ebee45e7f
# ╟─16c39f9f-08c5-44df-91a4-f6565c0e2145
# ╟─2dd83539-eb46-408a-a398-90303aeb1a94
# ╠═177d86ac-ce25-45c9-930c-95539f4778b8
# ╟─c7a111e8-4899-4f81-a10f-64ca9a0eed72
# ╟─a7d00915-e380-42b0-8a1a-bc653ff8010f
# ╟─d68285d6-719d-4d84-8a1b-6149c6acd8ea
# ╟─7954abc9-b9ed-4240-a7fb-26738eae9807
# ╟─f0d97466-3522-40b6-9b41-8a840dd4846c
# ╠═f6e77219-62c5-4e58-a197-7395095a664d
# ╟─5669704a-57ae-4db5-a37b-252f83ddd8a7
# ╟─136a316b-bab1-454d-bcfa-3854681f8b89
# ╠═0cec673a-5eb2-4554-b5a3-e2719bd977e1
# ╟─42f22aa5-df48-4d30-9185-df6efede8191
# ╠═25cb2a5d-6d2d-41a9-b7c6-bfa4e616c0f0
# ╟─7bf7aaa7-c005-4c61-b45b-33260147e8e1
# ╟─31270f76-ec06-4c15-a60e-497d5017c0dd
# ╟─72a1c126-6575-4375-adec-b1bc77d9854f
# ╠═dc83e29d-680a-49c2-91db-ab7af33661bb
# ╟─e600a641-9427-45ca-8e0c-6d2ccbeaa822
# ╠═3e2364ec-540f-4be3-bf9f-23369534a98b
# ╟─99946c17-c84c-44ca-bbd4-6075997a0d17
# ╟─2c16bb6c-d3af-47aa-8563-916cafbb2c49
# ╠═94895712-5ae0-49e7-8960-d081a6e92268
# ╟─fda1783b-e760-4e70-9d1c-0665674548f7
# ╟─b9b68558-e18e-47ce-843b-a71431738495
# ╟─71a82c6f-93fd-4c62-acca-898584960034
# ╟─f9f17fc1-3540-4dec-8fb4-939bf6e35310
# ╟─475384f5-4d81-4676-9878-b0489eca9781
# ╠═ddb2ee4c-3d11-48b1-9487-937a3efccfe5
# ╟─e04d2322-68e4-4355-9f1b-cb47d529769c
# ╟─6b4b4891-bfd9-4aa0-a8cd-b7662ffc2f8b
