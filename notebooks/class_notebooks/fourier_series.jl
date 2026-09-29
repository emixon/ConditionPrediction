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

# ╔═╡ 455ec274-03be-47e2-a582-016745b0f72e
begin
	if "PROJECT_DIR" ∈ keys(ENV)
		using Pkg; Pkg.activate(ENV["PROJECT_DIR"])
	end
	using PlutoUI

	using LinearAlgebra
	using Plots, LaTeXStrings
end

# ╔═╡ 42b3a8a8-1ae7-11ec-3a93-b5a9634a896e
md"""
# Fourier Series

#### Learning Objectives

After completing this worksheet, you should be able to:


* Explain the underlying mathematics of Fourier series.
* Calculate Fourier series coefficients ($a_k$ and $b_k$).
* Apply Fourier series methods to construct Fourier approximations.



"""

# ╔═╡ 41421d20-c4e6-42cd-bcf0-6c375cf0b911
PlutoUI.TableOfContents()

# ╔═╡ 86fb9ea8-b80c-4913-a79f-0f04115f9ebd
md"""
A [Fourier series](https://en.wikipedia.org/wiki/Fourier_series) is a periodic function composed of harmonically related sinusoids, combined by a weighted summation. With appropriate weights, one cycle (or period) of the summation can be made to approximate an arbitrary function in that interval (or the entire function if it too is periodic).

Mathematically,

$$f(x) = \frac{a_0}{2} + \sum_{k=1}^∞ a_k \cos \left(k \frac{2π x}{L} \right) + b_k \sin \left(k \frac{2π x}{L} \right)$$

where $$f(x)$$ is the function we want to approximate, $$L$$ is the length of the interval we want to approximate the function over, and $a_k$ and $b_k$ are constants. Also, $$a_0 = \sum{\left[ f(x) \right]} \frac{2 \mathrm{dx}}{L}$$.

Keep in mind that `x` is the independent variable, which will typically be time, so `L` will be the difference
between the first and last elements in the `x` vector (i.e. `x[end] - x[begin]`) and `dx` will
be the difference between any two elements of the `x` vector (i.e. `x[2]-x[1]`). Fourier series
only work when all the elements in the `x` vector are equally spaced, so it doesn't matter which
two elements you choose to calculate `dx`, because they should all be the same.

Note in the equation above that $k$ is inside the $\cos$ and $\sin$ functions, meaning that as $k$ increases, the period of the sine and cosine waves decreases.

In the equation above, we use values of $k$ from 1 to ∞, and when we do that we can reproduce our function **exactly**.

However, if use fewer than ∞ values of $k$, we can **approximately** reproduce our function $f(x)$. So, like the SVD, fourier series are a way to **change the coordinate** system that we are using to potentially allow a more compact representation of the patterns underlying a data set.



"""

# ╔═╡ 64dc8e07-1f3b-403a-a63f-bffef2016fbf
md"""
### Exercise 1

Create a function called `fourier_single` which takes scalar vales `k`, `aₖ`, and `bₖ`, and vector `x` and returns the k$^{\mathrm{th}}$ fourier series of `x`:

$$a_k \cos \left(k \frac{2π x}{L} \right) + b_k \sin \left(k \frac{2π x}{L} \right)$$
"""

# ╔═╡ 16f74ce0-9f0e-412e-8e8b-02db989c2548
function fourier_single(k, aₖ, bₖ, x::AbstractVector)
	L = x[end] - x[begin]
	return aₖ * cos.(k * 2π * x / L) + bₖ * sin.(k * 2π * x / L)
end

# ╔═╡ 73570551-d521-47ab-8ddc-6bd01d9f0578
md"""
## Example: Hat factory

Let's look at an example to build some intuition around this. We'll start with a function $f(x)$ that generates some data that kind of looks like a hat:

"""

# ╔═╡ 50450ff2-3315-45ab-a54a-19e289f18801
begin
	# Define domain
	dx = 0.001
	L = π
	x = collect(-1+dx:dx:1)*L
	n = length(x)
	nquart = Int(floor(n/4))

	# Define hat function
	f = 0*x;
	f[nquart:2*nquart] = 4 .* collect(1:nquart+1) ./ n
	f[2*nquart+1:3*nquart] = 1 .- 4 .* collect(0:nquart-1) ./ n

	plot(x, f, size=(400, 200), lab=:none, xlabel=L"x", ylabel=L"f(x)")
	# The syntax L"f(x)" etc. is enabled by the LaTeXStrings package and allows us
	# to write math on our plot using LaTeX.
end

# ╔═╡ 61c9f142-907b-4a31-a74d-fae8dadd0057
md"""
### Exercise 2

Let's try to reproduce this function with a Fourier series. The sliders below represent different values of $a_k$ and $b_k$. Move them around until you get a series that does a good job representing our hat function.

a₁ = $(@bind a₁ Slider(0.0:0.01:1.0, default=0.0, show_value=true)) ||
b₁ = $(@bind b₁ Slider(0.0:0.01:1, default=0.0, show_value=true))

a₂ = $(@bind a₂ Slider(0.0:0.01:1, default=0.0, show_value=true)) ||
b₂ = $(@bind b₂ Slider(0.0:0.01:1, default=0.0, show_value=true))

a₃ = $(@bind a₃ Slider(0.0:0.01:1, default=0.0, show_value=true)) ||
b₃ = $(@bind b₃ Slider(0.0:0.01:1, default=0.0, show_value=true))

a₄ = $(@bind a₄ Slider(0.0:0.01:1, default=0.0, show_value=true)) ||
b₄ = $(@bind b₄ Slider(0.0:0.01:1, default=0.0, show_value=true))

"""

# ╔═╡ a2fff103-3d82-4c33-8b3d-d324678466ec
if !ismissing(fourier_single(0.0, 1, 1, x))
	f1 = fourier_single(1.0, a₁, b₁, x)
	f2 = fourier_single(2.0, a₂, b₂, x)
	f3 = fourier_single(3.0, a₃, b₃, x)
	f4 = fourier_single(4.0, a₄, b₄, x)
	series_sum = f1.+f2.+f3.+f4.+sum(f)*dx/L

	plot(
		plot(x, [f1 f2 f3 f4], title="Fourier series"),
		plot(x, [f series_sum],
			title="Series sum", label=["x" "fourier"]),
		size=(600, 200),
	)
end

# ╔═╡ 16cada00-1440-4dff-8c4e-decf58dfa2b6
md"""
It's pretty difficult to get a good representation by hand, especially with only four Fourier modes, but hopefully you can get a sense of how it works, by sinusoidal waves with different frequencies and amplitudes either cancelling-out or reinforcing each other.

Luckily, there is an analytical method for doing this:

$$a_k = f(x) \cdot \cos \left(k \frac{2 π x}{L} \right) * \frac{2 \mathrm{dx}}{L}$$

$$b_k = f(x) \cdot \sin \left(k \frac{2 π x}{L} \right) * \frac{2 \mathrm{dx}}{L}$$

Note that "⋅" means the [dot product or inner product](https://en.wikipedia.org/wiki/Dot_product) of the two vectors on either side of it.

> 👉 Are there any similarities between these equations and the ones at the top of the worksheet?

### Exercise 3

Create two functions `calc_aₖ` and `calc_bₖ`, that take as arguments `k` (a scalar value) as well as `x` and `f` (vectors), and return the values of `aₖ` and `bₖ`, respectively.

"""

# ╔═╡ 49a04c92-316b-456d-9f53-b54e4e2f4949
function calc_aₖ(x::AbstractVector, f::AbstractVector, k::Number)
	L = x[end] - x[begin]
	dx = x[2] - x[1]
	ϕ = k * 2π * x / L
	return f ⋅ cos.(ϕ) * 2 * dx / L
end

# ╔═╡ 7f1c425b-9979-4c9b-a88f-6cfbc2f134fa
function calc_bₖ(x::AbstractVector, f::AbstractVector, k::Number)
	L = x[end] - x[begin]
	dx = x[2] - x[1]
	ϕ = k * 2π * x / L
	return f ⋅ sin.(ϕ) * 2 * dx / L
end

# ╔═╡ 4abe1360-910d-4a5a-affa-50ebc51f29c2
md"""
### Exercise 4

Now that we've created functions to calculate `aₖ` and `bₖ`, and to calculate a single Fourier series once we have those two values, now we can calculate the sum of `kmax` Fourier series:

$$\hat{f}(x) = \frac{a_0}{2} + \sum_{k=1}^\mathrm{kmax} a_k \cos \left(k \frac{2π x}{L} \right) + b_k \sin \left(k \frac{2π x}{L} \right)$$

Create a function `fourier_sum` that takes as arguments vectors `x` and `f` and scalar value `kmax` and returns the Fourier function approximation described by the equation above.

"""

# ╔═╡ d11a371d-9a6d-49f1-a168-451a4d085fac
function fourier_sum(x::AbstractVector, f::AbstractVector, kmax::Int)
	L = x[end] - x[begin]
	dx = x[2] - x[1]
	a₀ = sum(f) * 2 * dx / L
	fsum = sum([fourier_single(k, calc_aₖ(x, f, k), calc_bₖ(x, f, k), x) for k in 1:kmax])
	return a₀ / 2 .+ fsum
end

# ╔═╡ 70b1ee7b-1851-45cf-aaf7-4e5e79c133a2
md"""
Now we can calculate a Fourier approximation for our hat function for any arbitrary Fourier series number `k_hat`.

`k_hat` = $(@bind k_hat Slider(1:20, default=1, show_value=true))

"""

# ╔═╡ ef9e2dc8-e9e7-4ea4-aee2-9cc9b68dfc72
let
	fhat = fourier_sum(x, f, k_hat)
	if !ismissing(fhat)
		plot(x, [f, fhat], label=["x" "fhat"], size=(400,200))
	end
end

# ╔═╡ 5cc4b6a9-6dd7-4614-9138-8e3e06860ad6
md"""
## Amplitude and error

If we plot the values of `aₖ` and `bₖ` as a function of `k`, that gives us what's sometimes called the "power spectrum" or the "amplitude" of the signal at different frequencies.

Sometimes the `aₖ` and `bₖ` values are expessed as a complex number:

$$\mathrm{amplitude}_k = a_k + b_ki$$

For this particular function, the values of `bₖ` are all zeros, or more specifically within machine precision or "rounding error" of zero. This is a special case for this particular function, you can see a function below where this is not the case.

In the bottom plot, you can see that as $$\mathrm{k_{max}}$$ increases, the error between the original function and the fourier approximation monotonically decreases. For continuous functions, the error would equal zero if $$\mathrm{k_{max}}$$ equaled infinity.
"""

# ╔═╡ 5b1c08b2-a35e-4419-b730-e49ce228a64a
let
	if !ismissing(calc_aₖ([1,2],[1,2],1)) && !ismissing(fourier_sum([1,2],[1,2],1))
		n_amp = 100
		a_amp = [calc_aₖ(x, f, k) for k in 1:n_amp]
		b_amp = [calc_bₖ(x, f, k) for k in 1:n_amp]
		error = [norm(f .- fourier_sum(x, f, kmax)) for kmax in 1:n_amp]

		pa = plot(a_amp; yscale=:log10, ylabel="aₖ", lab=:none)
		pb = plot(b_amp; ylabel="bₖ", lab=:none,
				yformatter = yi -> round(yi, sigdigits=2))
		pe = plot(error; ylabel="error", xlabel=L"\mathrm{k_{max}}",
				lab=:none, yscale=:log10)
		plot(pa, pb, pe, layout=(3,1), size=(500, 400))
	end
end

# ╔═╡ d6078c6c-21f1-4738-8f29-9a803b8f2e46
md"""

For a function that has discontinuities (vertical lines), the Fourier series doesn't do quite as good of a job reproducing it. The oscillating or "ringing" behavior near the "corners" of the function below are called Gibbs phenomena, and you may see them around a lot in numerical simulation results.

You can see that even as you increase the value of `k` in this case, the ringing will never compeletely disappear.

`k_tophat` = $(@bind k_tophat Slider(1:200, default=100, show_value=true))

"""

# ╔═╡ 5678a3d6-44c0-4b39-891f-b0ddfc1dc4fb
let
	if !ismissing(calc_aₖ([1,2],[1,2],1)) && !ismissing(fourier_sum([1,2],[1,2],1))
		x = collect(0:0.01:10);

		f = zeros(size(x));
		f[floor(Int(floor(length(f)/4))):Int(floor(3*length(f)/4))] .= 1

		fFS = fourier_sum(x, f, k_tophat)

		p1 = plot(x, [f fFS], lab=[L"f(x)" L"\hat{f}(x)"])

		a_amp = [calc_aₖ(x, f, k) for k in 1:k_tophat]
		b_amp = [calc_bₖ(x, f, k) for k in 1:k_tophat]

		pa = plot(a_amp; ylabel="aₖ", lab=:none)
		pb = plot(b_amp; ylabel="bₖ", lab=:none)

		l = @layout [a{0.5w} grid(2,1)]

		plot(p1, pa, pb, layout=l, size=(680, 300))
	end
end

# ╔═╡ dd9183aa-cb67-45e0-8162-d9c1bc233b96
md"""

🎉 That's it!

Utility functions are below.

---

---

---

"""

# ╔═╡ dcbc5c16-e714-48c0-972d-d4f9825bd57f
begin
	almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]))

	still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]))

	keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]))

	yays = [md"Fantastic!", md"Splendid!", md"Great!", md"Yay ❤", md"Great! 🎉", md"Well done!", md"Keep it up!", md"Good job!", md"Awesome!", md"You got the right answer!", md"Let's move on to the next section."]

	correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]))

	hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))

	promising(text) = Markdown.MD(Markdown.Admonition("correct", "Looks promising!", [text]))

	not_defined(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define a variable called **$(Markdown.Code(string(variable_name)))**"]))

end

# ╔═╡ 0be713b5-0b4b-4e50-9566-42f168649cf1
if !@isdefined(fourier_single)
	not_defined(:fourier_single)
else
	let
		a = fourier_single(1, 0.5, 2.5, [-1, 2, 4, 6, 8])
		if ismissing(a)
			still_missing()
		elseif !isa(a, Vector)
			keep_working(md"the result should be a vector")
		elseif length(a) != 5
			keep_working(md"the result should have the same number of elements as `x`")
		elseif norm(a) ≈ 3.872983346207417
			keep_working(md"Have you correctly calculated `L`?")
		elseif !(norm(a) ≈ 3.933650786006321)
			keep_working(norm(a))
		else
			correct()
		end
	end
end

# ╔═╡ 038eb79c-1d9e-44bc-83fa-a20170bfde68
if !@isdefined(series_sum)
	not_defined(:series_sum)
else
	let
		a = series_sum
		if ismissing(a)
			still_missing()
		elseif norm(series_sum .- f, 1) / sum(f) > 0.4
			keep_working()
		else
			correct()
		end
	end
end

# ╔═╡ d6c2e67e-fd35-4706-9c91-61aa46a43ee3
if !@isdefined(calc_aₖ)
	not_defined(:calc_aₖ)
else
	let
		a = calc_aₖ([0, 2, 4, 6, 8], [1,0,1,2,1], 3)
		if ismissing(a)
			still_missing()
		elseif !isa(a, Number)
			keep_working(md"the result should be a number")
		elseif !(a ≈ 0.5)
			keep_working()
		else
			correct()
		end
	end
end

# ╔═╡ 56f72381-29fc-4c19-9083-a4e0794370ed
if !@isdefined(calc_bₖ)
	not_defined(:calc_bₖ)
else
	let
		a = calc_bₖ([0, 2, 4, 6, 8], [1,0,1,2,1], 3)
		if ismissing(a)
			still_missing()
		elseif !isa(a, Number)
			keep_working(md"the result should be a number")
		elseif !(a ≈ 1.0)
			keep_working()
		else
			correct()
		end
	end
end

# ╔═╡ f5f506f3-80e2-4105-9011-4df559f27f63
if !@isdefined(fourier_sum)
	not_defined(:fourier_sum)
else
	let
		a = fourier_sum([0, 2, 4, 6, 8], [1,2,3,2,1], 4)
		if ismissing(a)
			still_missing()
		elseif !isa(a, Vector)
			keep_working(md"the result should be a vector")
		elseif length(a) != 5
			keep_working(md"the result should have the same number of elements as `x` and `f`")
		elseif !(norm(a) ≈ 14.97706580075016)
			keep_working()
		else
			correct()
		end
	end
end

# ╔═╡ Cell order:
# ╟─42b3a8a8-1ae7-11ec-3a93-b5a9634a896e
# ╠═455ec274-03be-47e2-a582-016745b0f72e
# ╟─41421d20-c4e6-42cd-bcf0-6c375cf0b911
# ╠═86fb9ea8-b80c-4913-a79f-0f04115f9ebd
# ╟─64dc8e07-1f3b-403a-a63f-bffef2016fbf
# ╠═16f74ce0-9f0e-412e-8e8b-02db989c2548
# ╟─0be713b5-0b4b-4e50-9566-42f168649cf1
# ╟─73570551-d521-47ab-8ddc-6bd01d9f0578
# ╠═50450ff2-3315-45ab-a54a-19e289f18801
# ╟─61c9f142-907b-4a31-a74d-fae8dadd0057
# ╟─a2fff103-3d82-4c33-8b3d-d324678466ec
# ╟─038eb79c-1d9e-44bc-83fa-a20170bfde68
# ╟─16cada00-1440-4dff-8c4e-decf58dfa2b6
# ╠═49a04c92-316b-456d-9f53-b54e4e2f4949
# ╟─d6c2e67e-fd35-4706-9c91-61aa46a43ee3
# ╠═7f1c425b-9979-4c9b-a88f-6cfbc2f134fa
# ╟─56f72381-29fc-4c19-9083-a4e0794370ed
# ╟─4abe1360-910d-4a5a-affa-50ebc51f29c2
# ╠═d11a371d-9a6d-49f1-a168-451a4d085fac
# ╟─f5f506f3-80e2-4105-9011-4df559f27f63
# ╟─70b1ee7b-1851-45cf-aaf7-4e5e79c133a2
# ╠═ef9e2dc8-e9e7-4ea4-aee2-9cc9b68dfc72
# ╟─5cc4b6a9-6dd7-4614-9138-8e3e06860ad6
# ╠═5b1c08b2-a35e-4419-b730-e49ce228a64a
# ╟─d6078c6c-21f1-4738-8f29-9a803b8f2e46
# ╠═5678a3d6-44c0-4b39-891f-b0ddfc1dc4fb
# ╟─dd9183aa-cb67-45e0-8162-d9c1bc233b96
# ╟─dcbc5c16-e714-48c0-972d-d4f9825bd57f
