### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ a3f1a69a-18be-11ec-0bb9-cf9491ee110c
begin
	if "PROJECT_DIR" ∈ keys(ENV)
		using Pkg; Pkg.activate(ENV["PROJECT_DIR"])
	end
	using PlutoUI
	using MAT
	using FFTW
	using Plots
end

# ╔═╡ 9399e7a6-21c4-4cc3-a939-c129aca986a7
md"""
# The Fast Fourier Transform (FFT)


#### Learning Objectives

After completing this worksheet, you should be able to:



* Explain the calculation process and physical interpretation of the Fast Fourier Transform (FFT).

* Apply the Fast Fourier Transform to calculate the frequency components of a dataset.





The Fast Fourier Transform is a computationally efficient way to calculate the Discrete Fourier Transorm of a dataset.

We will be practicing how to do this using data from an experiment
[(Report on the Building Structural Health Monitoring Problem Phase 2)](https://authors.library.caltech.edu/records/1k7mb-x6m79) testing the integrity of a structure while it is being shaken.

"""

# ╔═╡ 42dbc751-0c75-415a-a162-3085d7e6f93e
md"""
Today we're going to use the FFT to examine the frequencies of the vibrations that this structure undergoes.
"""

# ╔═╡ ebb2ba3e-9095-470b-8f86-16dec6957ab7
PlutoUI.TableOfContents()

# ╔═╡ b1c23be2-cd8a-41a8-b1a7-8916015ba6c1
md"""
## But first... Mechanics of the FFT

Imagine that we have a signal and we want to analyze its power spectrum. We will make up a simple signal to start with, which is just the composition of cosine waves at 1 and 30 hz and a sine wave at 10 hz:

"""

# ╔═╡ 66ed463e-49c1-492b-acec-977e9a591b01
function signal(n, samplerate)
	f1 = 1 # hz
	f2 = 10 # hz
	f3 = 30 # hz
	times = (0:(n-1))/samplerate
	signal = cos.(times*f1*2π) + sin.(times.*f2*2π) + cos.(times*f3*2π)
	return times, signal
end

# ╔═╡ 8e06b15d-2da6-41b7-9c7f-f35f22658571
md"We'll assume that the signal amplitude represents a distance traveled in mm."

# ╔═╡ 0581bead-b717-4e7f-8081-b38c1be354c7
begin
	examplesamplerate = 100 # Samples per second or hertz
	exampletimes, examplesignal = signal(1000, examplesamplerate)
	plot(exampletimes, examplesignal, size=(400,200),
		xlabel="time (s)", ylabel="distance (mm)", lab=:none)
end

# ╔═╡ 6a21c49e-2ff9-4dfa-9a10-2090c8856ba6
md"""
It is straightforward to calculate the FFT of this signal using the function [`fft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#Public-Interface-1):

"""

# ╔═╡ 85b3a1d6-8166-4be8-9e35-db5ff4966841
example_amplitudes = fft(examplesignal)

# ╔═╡ 03e2712e-402b-494e-a4f6-ea86712b1d3e
md"""
...and we can also calculate the frequency of each index in the vector returned by the `fft` function using the function [`fftfreq`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fftfreq):
"""

# ╔═╡ ab385445-9022-45df-b539-5074aefbb311
example_freqs = fftfreq(length(exampletimes), examplesamplerate)

# ╔═╡ 0aadf828-2c42-4c8f-81d3-ca9c7dc52c25
md"""We can plot the result..."""

# ╔═╡ 4af962fb-a6ba-4310-bd18-ee6aba60259c
begin
	plot(example_freqs, real(example_amplitudes), label="Real component")
	plot!(example_freqs, imag(example_amplitudes), label="Imaginary component")
	plot!(size=(400,200))
end

# ╔═╡ f8517650-ad6c-4aa8-9f85-c7a3d7e6dd20
md"""...but that doesn't look right!

If you look at the `example_amplitudes` data above, you can see that each value in the vector is a complex number, meaning that it has both a real and an imaginary part. This allows the result to contain information both about the amplitude of a signal at a given frequency as well as its phase—how much it's made up of a sine vs. cosine wave.

However, for applications in this class we only care about the overall amplitude, so we can take the absolute value of our fft result:
"""

# ╔═╡ 4c74b710-17b5-433d-938f-e0b7c42acdd7
example_amplitudes_abs = abs.(example_amplitudes)

# ╔═╡ 1a521159-769c-4c9e-ab21-376f14c7a1cf
md""" You can also take the square root of the vector multiplied by its complex conjugate as they do in the book, it's the same thing:
"""

# ╔═╡ 8924ce06-3e47-4d8c-84c2-4f35c3c151c7
sqrt.(example_amplitudes .* conj.(example_amplitudes))

# ╔═╡ 214ceccf-6a10-4ae9-a6e9-67d45bc8a43b
md"""Let's see what it looks like when we plot that:"""

# ╔═╡ 77eb03d4-d11b-4ef1-b067-a67680781237
plot(example_freqs, example_amplitudes_abs, lab=:none,
	xlabel="Frequency (hz)",
	size=(400,200),
)

# ╔═╡ 7b816c8f-ced8-4904-88d4-0b9592c1149b
md"""
This still doesn't look quite right! Now we have negative frequencies, and we also have a horizontal line at y=0.

We can fix the second problem using the [`fftshift`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fftshift) function to switch the first and second halves of the data so that we don't get the horizontal line anymore:

(The two plots look the same, so you'll just have to trust me that there's an extra horizontal line at y=0 in the first one but not the second one.)
"""

# ╔═╡ a5680ed1-53e9-4ae3-9937-d4fb9eac8d7a
plot(fftshift(example_freqs), fftshift(example_amplitudes_abs), lab=:none,
	xlabel="Frequency (hz)",
	size=(400,200),
)

# ╔═╡ e1107b4e-a81f-48a5-bc31-da09225c5fbd
md"""
However, the negative frequencies are still there. It turns out that this happens [because](https://dsp.stackexchange.com/a/449) the DFT and FFT are computed using complex spirals instead of the sinusoids we used in the previous class and the negative freqencies therefore hold information regarding the phase of the signal and are necessary if we want to reconstruct it.

For the purposes of this class, however, we can just note that the amplitudes for the positive and negative frequencies are mirrors of each other, and just discard the negative frequency part:

"""

# ╔═╡ 4daf3ccd-2f6b-4f1f-a1b8-b13c02853099
begin
	positivefreqs = example_freqs .> 0 # We can also do ">= 0" if we want to keep the zero frequency

	plot(
		example_freqs[positivefreqs],
		example_amplitudes_abs[positivefreqs],
		lab=:none,
		xlabel="Frequency (hz)",
		size=(400,200),
	)
end

# ╔═╡ ea42176d-3137-41d4-a926-95a8890237ef
md"""
In the plot above, you can see that we have three amplitude spikes a 1 hz, 10 hz, and 30 hz, corresponding to the three sine or cosine waves that make up our original signal.

However, the y-axis values are still a little strange. In the result of the FFT, each spike represents the *number* of samples at that frequency. To instead show the average amplitude at that frequency, we can divide by the effective number of samples, which is the number of samples in the original time series divided by two (because half of the frequencies in the FFT result are negative and therefore discarded).

Once we do that, the amplitude of each spike is equal to one, which is the same as our original signal.
"""

# ╔═╡ 04c912af-3579-424c-9bc0-a53ebd701b55
	plot(
		example_freqs[positivefreqs],
		example_amplitudes_abs[positivefreqs] / length(examplesignal) * 2,
		lab=:none,
		xlabel="Frequency (hz)", ylabel="Amplitude (mm)",
		size=(400,200),
	)

# ╔═╡ ce868a86-267e-4b52-b9c2-09e8275565be
md"""
## Structural vibrations
"""

# ╔═╡ 50a5bdad-d095-48a6-b026-db8ef3192586
md"""
## Load

The original data is in MATLAB format. We can read MATLAB files using the MAT.jl package:
"""

# ╔═╡ 184644b3-6a30-45d6-8175-9ea5c028e7f0
begin
	filename = "/home/julia/project/data/shm01ss1.mat"
	file = matopen(filename)
	names(file)
end

# ╔═╡ 26e96e27-280a-455d-b956-16984dcd9042
md"""
The two items we're most interested in in this file are `dasy`, which is the data itself (dasy is the name of the insturement measuring the vibrations), and `fsdasy`, which is the instrument sample rate, presumably in hertz.
"""

# ╔═╡ 7b406030-2284-4a43-9e27-740d294dd1bd
fsdasy = read(file, "fsdasy")

# ╔═╡ 070a837f-fbf7-43a4-b6be-ac49c32e6dce
dasy = read(file, "dasy")

# ╔═╡ aeab136d-7a0c-4cd0-a35a-3a0a8796e6d7
md"""
We can see above that the `dasy` data is a dictionary where each key is the name of a sensor (`DA11` etc.) and each value associated with the key is a time series of measurements.
"""

# ╔═╡ b9b27455-c45c-498c-bebe-e6a9fc35fbe4
md"""
## Clean

We can calculate our sample times by dividing the sample numbers by the sample frequency:

"""

# ╔═╡ cf258be1-9fc2-4d58-9b72-e5d55c406785
times = (1:length(dasy["DA11"][:])) / fsdasy

# ╔═╡ eaf5023f-a368-491f-8565-9d09fe3a3bb9
md"""
## Visualize

Next, let's visualize our dataset by making a series of time-series plots, one for each sensor.

"""

# ╔═╡ 29636573-85b4-415c-8df5-d6364f1df9fe
begin
	sensor_order = ["DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07",
		"DA08", "DA09", "DA10", "DA11", "DA12", "DA13", "DA14", "DA15", "DA16"]
	plots = []
	for (i, sensor) in enumerate(sensor_order)
		p = plot(times, dasy[sensor], lab=:none, title=sensor)
		if i%4 == 1
			ylabel!("Displacement (mm)")
		end
		if (i-1) ÷ 4 == 3
			xlabel!("Time (s)")
		else
			xticks!(:none)
		end
		push!(plots, p)
	end
	plot(plots..., layout=(4,4), shared=:both, size=(800, 800))
end

# ╔═╡ eca50045-5ab6-4ab5-91ea-c357e623e4dd
md"""

> 👉 What do these plots show us?

"""

# ╔═╡ 47d785a2-ab3b-4da4-acde-6ab003bf01b5
md"""
## Analyze

Lets make another set of plots, showing the power spectral density (PSD) of each sensor. A PSD plot has the frequencies of the Fourier series in the x-axis, and the amplitudes in the y-axis.

"""

# ╔═╡ deef42c7-7a27-4df9-8764-aaec1eb29d27
md"""
### Exercise 1a: FFT

Create a function called `calc_fft` that takes a vector as its argument and returns the freqencies (excluding the negative and zero frequencies) and normalized amplitudes of the FFT transform of that vector.

"""

# ╔═╡ a8e98062-729c-4d09-9abb-0aad9fec898a
function calc_fft(a::AbstractVector, samplerate)
	freqencies = fftfreq(length(a), samplerate)
	pos_freqencies = freqencies .> 0
	abs_amplitudes = abs.(fft(a))
	norm_amplitudes = abs_amplitudes[pos_freqencies] / length(a) * 2
	return freqencies[pos_freqencies], norm_amplitudes
end

# ╔═╡ cf11c6b1-adbc-46c3-9ea8-43045d45c3a0
md"""
### Exercise 1b: PSD plots

Create a PSD plot for each sensor.

"""

# ╔═╡ 648433a3-d896-4172-8e4c-4a56bbf66c72
begin
	psd_plots = []
	for (i, sensor) in enumerate(sensor_order)
		sensor_fft = calc_fft(dasy[sensor][:], 200)

		p = plot(sensor_fft[1],sensor_fft[2],lab=:none,xlabel="Frequency (hz)",size=(400,200))
		push!(psd_plots, p)
	end
	psd_plot = plot(psd_plots..., layout=(4,4), shared=:both, size=(800, 800))
end

# ╔═╡ 2f00c0c0-1aa1-4abd-b20d-c337c6878026
md"""

> 👉 What can we learn from these plots that we couldn't see in the time series?

"""

# ╔═╡ caaec4fe-1a56-4859-8107-8a4fde573f82
md"""

🎉 That's it!

Utility functions are below.

---

---

---

"""

# ╔═╡ a7496346-9c49-4282-b2a3-dfc81a1a2b1b
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

# ╔═╡ 34b74933-84ef-4773-8162-86fcc58a47f9
if !@isdefined(calc_fft)
	not_defined(:calc_fft)
else
	let
		r = calc_fft([2,0,2,0,2,0,1], 1)
		if ismissing(r)
			still_missing()
		elseif length(r) != 2
			keep_working(md"your function should return 2 variables!")
		else
			f, a = r
			if !isa(f, Vector) || !isa(a, Vector)
				keep_working(md"your frequencies and amplitudes should be Vectors")
			else
				if length(a) != 3 || length(f) != 3
					keep_working(md"the number of frequencies or amplitudes isn't quite right")
				elseif !(sum(f) ≈ 0.8571428571428571) || !(sum(a) ≈ 1.7476632068585871)
					keep_working()
				else
					correct()
				end
			end
		end
	end
end

# ╔═╡ fc97fa9d-9ebd-44bb-b0f1-cf8fe3920706
if !@isdefined(psd_plot)
	not_defined(:psd_plot)
else
	let
		p = psd_plot
		if ismissing(p)
			still_missing()
		elseif !isa(p, Plots.Plot)
			keep_working(md"`psd_plot` is not a plot!")
		else
			if length(p.subplots) != 16
				keep_working(md"Your plot should have 16 subplots.")
			elseif !(sum(p[4][1][:x]) ≈ 1.49995e6)
				keep_working()
			elseif !(sum(p[4][1][:y]) ≈ 0.003949974262360918)
				keep_working()
			elseif !(sum(p[10][1][:x]) ≈ 1.49995e6)
				keep_working()
			elseif !(sum(p[10][1][:y]) ≈ 0.023722120820886206)
				keep_working()
			else
				correct()
			end
		end
	end
end

# ╔═╡ Cell order:
# ╟─9399e7a6-21c4-4cc3-a939-c129aca986a7
# ╟─42dbc751-0c75-415a-a162-3085d7e6f93e
# ╠═a3f1a69a-18be-11ec-0bb9-cf9491ee110c
# ╟─ebb2ba3e-9095-470b-8f86-16dec6957ab7
# ╟─b1c23be2-cd8a-41a8-b1a7-8916015ba6c1
# ╠═66ed463e-49c1-492b-acec-977e9a591b01
# ╟─8e06b15d-2da6-41b7-9c7f-f35f22658571
# ╠═0581bead-b717-4e7f-8081-b38c1be354c7
# ╟─6a21c49e-2ff9-4dfa-9a10-2090c8856ba6
# ╠═85b3a1d6-8166-4be8-9e35-db5ff4966841
# ╟─03e2712e-402b-494e-a4f6-ea86712b1d3e
# ╠═ab385445-9022-45df-b539-5074aefbb311
# ╟─0aadf828-2c42-4c8f-81d3-ca9c7dc52c25
# ╠═4af962fb-a6ba-4310-bd18-ee6aba60259c
# ╟─f8517650-ad6c-4aa8-9f85-c7a3d7e6dd20
# ╠═4c74b710-17b5-433d-938f-e0b7c42acdd7
# ╟─1a521159-769c-4c9e-ab21-376f14c7a1cf
# ╠═8924ce06-3e47-4d8c-84c2-4f35c3c151c7
# ╟─214ceccf-6a10-4ae9-a6e9-67d45bc8a43b
# ╠═77eb03d4-d11b-4ef1-b067-a67680781237
# ╟─7b816c8f-ced8-4904-88d4-0b9592c1149b
# ╠═a5680ed1-53e9-4ae3-9937-d4fb9eac8d7a
# ╟─e1107b4e-a81f-48a5-bc31-da09225c5fbd
# ╠═4daf3ccd-2f6b-4f1f-a1b8-b13c02853099
# ╟─ea42176d-3137-41d4-a926-95a8890237ef
# ╠═04c912af-3579-424c-9bc0-a53ebd701b55
# ╟─ce868a86-267e-4b52-b9c2-09e8275565be
# ╟─50a5bdad-d095-48a6-b026-db8ef3192586
# ╠═184644b3-6a30-45d6-8175-9ea5c028e7f0
# ╟─26e96e27-280a-455d-b956-16984dcd9042
# ╠═7b406030-2284-4a43-9e27-740d294dd1bd
# ╠═070a837f-fbf7-43a4-b6be-ac49c32e6dce
# ╟─aeab136d-7a0c-4cd0-a35a-3a0a8796e6d7
# ╟─b9b27455-c45c-498c-bebe-e6a9fc35fbe4
# ╠═cf258be1-9fc2-4d58-9b72-e5d55c406785
# ╟─eaf5023f-a368-491f-8565-9d09fe3a3bb9
# ╠═29636573-85b4-415c-8df5-d6364f1df9fe
# ╟─eca50045-5ab6-4ab5-91ea-c357e623e4dd
# ╟─47d785a2-ab3b-4da4-acde-6ab003bf01b5
# ╟─deef42c7-7a27-4df9-8764-aaec1eb29d27
# ╠═a8e98062-729c-4d09-9abb-0aad9fec898a
# ╟─34b74933-84ef-4773-8162-86fcc58a47f9
# ╟─cf11c6b1-adbc-46c3-9ea8-43045d45c3a0
# ╠═648433a3-d896-4172-8e4c-4a56bbf66c72
# ╟─fc97fa9d-9ebd-44bb-b0f1-cf8fe3920706
# ╟─2f00c0c0-1aa1-4abd-b20d-c337c6878026
# ╟─caaec4fe-1a56-4859-8107-8a4fde573f82
# ╟─a7496346-9c49-4282-b2a3-dfc81a1a2b1b
