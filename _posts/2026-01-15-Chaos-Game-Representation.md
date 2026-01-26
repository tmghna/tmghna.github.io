---
layout: post
title: Unveiling Paradigms with Chaos Game Representation
description: For the purpose of Meraki 26' (Magazine)
authors: [Tamaghna, Vedant]
categories: [General]
tags: [fractal, chaos game, k-mer, PCA, clustering, CNN, signal processing]
---

## Introduction

For decades, the “gold standard" of bioinformatics has been genomic sequence alignment, be it through local tools like BLAST or global aligners such as Needleman-Wunsch; the core objective is to find a basepair-to-basepair correspondence between two or more sequences. These programs assume that biological sequences share regions of similarity in the same linear order.

However, modern genomics has exposed several critical downsides to this approach:

- **Deviation from collinearity**: Viral and bacterial genomes often undergo large-scale rearrangements, shuffling, and horizontal gene transfers that violate the assumption of linear order.
- **The Twilight Zone**: Defined as the point where the signal-to-noise ratio in genetic data becomes dangerously low, it describes a specific range of sequence similarity where traditional alignment tools can no longer reliably distinguish between truly related sequences (homologs) and random noise.
- **Computational Intensity**: Calculating a mathematically optimal alignment becomes an *NP-hard* problem. However, a standard pairwise sequence alignment method, such as Needleman-Wunsch, is better, but it also has a space and time complexity of $$O(nm)$$ (with *n* and *m* being the lengths of the two sequences to be compared), which becomes significant with increasing sequence length.

To bypass these bottlenecks, researchers have developed alignment-free methods. These techniques quantify similarity without attempting to line up individual bases. The most visually and mathematically distinct among these is **Chaos Game Representation (CGR)**.

> **NP (Nondeterministic Polynomial) time is a concept within computational complexity theory. Think of NP-Hard problems as problems where finding the answer is difficult, but verifying a given answer is easy. For example, a Sudoku puzzle! Solving it takes time, but checking if a completed one is correct is nearly instant.*
> 

## So how does CGR actually work?

It transforms a data sequence (in this context, let’s say a nucleotide sequence from a FASTA file) into a 2D Fractal Image using an iterated function system (IFS):

1. Assign Vertices: A square is defined where each corner represents one of the four nucleotides (A, C, G, T). In this article, a horizontal order of purines (and likewise pyrimidines) has been established (A-T/G-C). However, a diagonal order of the same can also be found in some literature. 
2. Iterative Plotting: Starting from the center, a pointer moves halfway toward the vertex corresponding to the first base in the sequence and plots a point.
3. Recursive Geometry: This process repeats for every base in the sequence, always moving halfway from the current position toward the next vertex.

The compact notation for IFS:

$$
\begin{equation}
P_i = P_{i-1} \ + \
s_f \cdot \big(V_{i-1} - P_{i-1}\big)
\end{equation}\tag{1}
$$

Where $$s_f$$ is the scaling factor; $$1/2$$ for DNA. $$V_{i-1}$$  is the vertex in the square corresponding to $$(i-1)^{th}$$ nucleotide.
This system defines a contractive IFS, so the orbit converges to a compact invariant set (known as the attractor).

NOTE: The pattern formation in CGR does not necessarily imply biological structure. It depends upon the polygon geometry and contraction ratio (which is $$1/2$$ here). For example, in an equilateral triangle, even purely random vertex selections produce the *Sierpiński gasket*, and *star-like fractals* are formed for regular polygons of 5 or 6 vertices. It is only for a square when random symbol selection doesn’t produce a fractal pattern.

**So, when can patterns appear for the square?**
Patterns do appear in square-based CGR only if the sequence is *not* random. That is, if the sequence contains symbol bias, forbidden transitions, or some Markov dependence. Jeffrey was the first to apply CGR to DNA. Instead of using a triangle, the CGR was based on a square, with the four vertices representing the four nucleotides, resulting in an image that can be interpreted as the “genomic fingerprint”.

## Frequency Chaos Game Representation

While the original CGR uses exact coordinates for each point, a discretisation called the frequency chaos game representation (FCGR) enables a coarse-grained and less noisy CGR abstraction for sequences, which makes implementation of learning-model training super easy.

The plotting of a classical CGR on a computer screen inevitably results in image compression. Hence, researchers introduced the FCGR methodology by separating the CGR for each sequence with a grid in resolutions of $$2^k \times 2^k$$ , in order to count the frequencies of different *k-mers*. Each k-mer is uniquely mapped to a pixel via a recursive chaos game algorithm, and its frequency is obtained by counting occurrences as the sequence is scanned with a sliding window of length k.

> It is often suggested to use intra-sample scaling for a particular sequence as $$f_i =\frac{\text{count}(k\text{-mer}_i)}{N-k+1}$$ to avoid the “genome-size effect”. A few papers also suggest inter-sample scaling by the use of Z-score normalisation to ensure a biologically “rare” k-mer is not overshadowed by a “common” k-mer.
> 

The formation of FCGR image vector can be explained by the following pseudo-code:

```
// Define nucleotide-to-vertex mapping
MAP ← { 'A':(0,0), 'T':(0,1), 'G':(1,0), 'C':(1,1) }

Function get_FCGR ← Parameters(SEQUENCE as S, k):
    a. Array G ← size:(2^k × 2^k) with zeros
    b. For each k-mer w in S: // The sliding window
          i. Initialize (x, y) ← (0.5, 0.5)
         ii. For each nucleotide c in w:
                (vx, vy) ← MAP[c]
                x ← x + 0.5 * (vx − x)
                y ← y + 0.5 * (vy − y)
        // Discretization
        iii. Convert (x, y) to grid indices (i, j)
         iv. G[i, j] ← G[i, j] + 1
    // Intra-sample normalization
    c. Normalize G by dividing by total number of k-mers
    d. Vector f ← Flatten G
    e. Return f
```

The image vector  $$\mathbf{f} = (f_1, f_2, \dots , f_{4^k})$$ , where $$f_i$$ is the normalised frequency of the i-th k-mer. This vector resides in a $$4^k$$-dimensional space, and the CGR image is merely a 2D representation of it. 

> FCGR is used in machine learning tasks for bioinformatics involving Species characterisation, Phylogeny, Anomaly detection, and Domain classification, to name a few.
> 

A data matrix $$\mathbf{X}$$ can now be generated where each row is a flattened FCGR image vector. The dimension of the matrix, for $$n$$ sequences each with a k-mer of size $$k$$, will then become $$n\times 4^k$$ 

$$
\begin{equation}
\mathbf{X}
=
\begin{bmatrix}
\mathbf{f_1} \\
\mathbf{f_2} \\
\vdots \\
\mathbf{f_n}
\end{bmatrix}\tag{2}
\end{equation}
$$

Formation of this matrix is fairly simple in python by reading a multi-sequence FASTA file and passing it to the FCGR function:

```python
from Bio import SeqIO

def read_fasta(fasta_file_path): # Read sequences from FASTA file
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    return sequences
    
k = 4 # K-mer size (dimensionality = 4^4 = 256)
sequences = read_fasta("FASTA file path")
X = np.array([get_fcgr(seq, k) for seq in sequences])
```

Sophisticated statistical tools, such as Principal Component Analysis (PCA), are employed to reduce the enormous dimensionality of the feature vector.  

This is done by first calculating the mean-centred matrix $$\mathbf{B}=\mathbf X-\mathbf {\bar X}_{avg}$$, followed by the formation of a covariance matrix $$\mathbf C$$. This matrix illustrates how variables change together, with diagonal elements representing variances and off-diagonal elements representing covariances. Instead of calculating a $$4^k\times4^k$$ covariance matrix $$\mathbf C = \mathbf B^T \mathbf B$$, $$\mathbf B$$ is decomposed directly using Single Value Decomposition (SVD) method. 

It is then followed by identifying the eigenvectors (Principal components) that capture the maximum variance. The eigenvectors corresponding to the largest eigenvalues become the principal components, highlighting the most essential patterns, whereas, low eigenvalues usually represent noise or minor sequence variations that don't help in distinguishing viral families. In FCGR, typically enough components are retained to explain **80-95%** of the variance. A *scree plot* is generated to visualise the variances along different PC axes.

Clustering strategies such as K-means clustering and Hierarchical clustering can then be incorporated into the manageable set of PCA-reduced data.

## Breakthroughs

The time complexity of FCGR construction comprises two parts: 

1. Sliding window over the sequence, which is linear $$N-k+1=O(N)$$
2. Iteration over k bases, where each step has constant time, hence cost per k-mer $$O(k)$$

Total time complexity henceforth $$= O(Nk) = O(N)$$ as k is typically small (4-8 in practice). This represents a significant leap from a standard pairwise sequence alignment method, which has a quadratic time complexity, as mentioned earlier.

Recently, convolutional neural networks (CNNs) have been trained on the feature vectors extracted from FCGR representation, and this has been shown to outperform traditional ML classifiers. In 2023, researchers used FCGR images with ResNet50 (a deep CNN architecture with 50 layers) to classify SARS_CoV-2 sequences into clades with **96.29%** accuracy, surpassing conventional tools like Covidex.

In another study, the transition from global alignment-free metrics to local detection represents a significant breakthrough. The alignment-free local homology (**alfy**) model tackles this transition by utilising *shustrings* (Shortest Unique Substrings) to calculate exact match lengths at every position in a sequence. It demonstrates that, in clinical benchmarks for HIV-1 subtyping, the model performs comparably to high-precision tools like SCUEAL in terms of accuracy, while also increasing speed by up to 60,000 times. (From 6.9 hours down to 0.4 seconds). Another major advantage is the model’s ability to detect genomic segments that have no homologs in other strains. This led to the discovery of a specific O1-antigen gene cluster in avian *E. coli,* which is nearly identical to one found in human clinical isolates, proving that poultry pathogens can be transmitted to humans.

> **Analogy for ease of understanding: Imagine your DNA as a very long book made of 4 letters (A,T,G,C). Now the task at hand is comparing two books and lining them up page by page to see if they are the same, but this takes a long time if the books are large. What shustrings do is look at one letter and ask, “What is the shortest secret word I cannot find in other books?” If the secret word is very long, the books are almost the same in that spot; if they are very short, the books are very different.*
> 

## Beyond the realm of bioinformatics

Chaos game representation has a wide variety of applications, even beyond the realm of bioinformatics. 

- **Malware Detection**: Just as species have genomics signatures, malware files such as *Trojans* or *Ransomware* have a structural signature hidden in the byte sequences of .exe or .elf files, which when converted into FCGR, allows a CNN to classify malware ultrafast without *sandboxing* the file.
- **Traffic Analysis**: Identifying the type of traffic (e.g., VPN vs Tor) is troublesome when data is encrypted. Studies have shown that through the application of CGR to Intrusion Detection Systems (IDS) by mapping IP packet headers to a fractal image, the system can detect DDoS attacks as a sudden shift in the time patterns for the *chaos density* of the plot.
- **Biometrics**: Multiple novel methods for fingerprint and iris recognition have been presented, which compare two CGR images using a simple Euclidean distance, a pre-trained CNN, or by calculating the fractional dimension associated with the fractal image using the box-counting method. These are significantly faster than the computationally expensive point-wise matching of traditional biometric techniques.

CGR also has applications in signal processing across various fields, including finance, cardiology, and seismology, among others, making it the go-to standard as a universal sequence analyser. Potential use cases are also unending, with possible applications in diagnostics, and many more. The integration of CGR with modern ML marks a paradigm shift in how we process high-dimensional sequential data, making the idea of CGR forever recognised as a crucial framework in the academic community, connecting threads across multiple domains of science, learning to read the very texture of information itself. This is just the tip of the iceberg; who knows where this chaos may lead us?

But wait! Was the “Chaos” a misnomer all along?

## References

| S No | Paper Name                                                                                                                                                                    | URL                                                                                                         |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 1    | **Löchel et al. (2021)**. *Chaos game representation and its applications in bioinformatics*                                                                                  | https://doi.org/10.1016/j.csbj.2021.11.008                                                                  |
| 2    | **Jeffrey, H.J. (1990)**. *Chaos game representation of gene structure*.                                                                                                      | https://doi.org/10.1093/nar/18.8.2163                                                                       |
| 3    | **Deschavanne et al. (1999)**. *Genomic Signature: Characterization and Classification of Species Assessed by Chaos Game Representation of Sequences*                         | https://doi.org/10.1093/oxfordjournals.molbev.a026048                                                       |
| 4    | **CGRclust:** *Chaos Game Representation for Twin Contrastive Clustering of Unlabelled DNA Sequences*                                                                         | https://doi.org/10.1186/s12864-024-11135-y                                                                  |
| 5    | **Burma et al. (1992)**. *Genome analysis: A new approach for visualization of sequence organization in genomes*                                                              | https://doi.org/10.1007/BF02720095                                                                          |
| 6    | **Domazet-Lošo et al. (2011)**. *Alignment-free detection of local similarity among viral and bacterial genomes*                                                              | [10.1093/bioinformatics/btr176](https://doi.org/10.1093/bioinformatics/btr176)                              |
| 7    | **Jampour et al. (2009)**. *A new fast technique for fingerprint identification with fractal and chaos game theory*                                                           | https://doi.org/10.1142/S0218348X10005020?urlappend=%3Futm_source%3Dresearchgate.net%26utm_medium%3Darticle |
| 8    | Diaconis & Freedman (1999), *Iterated random functions*                                                                                                                       |                                                                                                             |
| 9    | **Bimal K Sarkar et al. (2021)**. *Determination of k-mer density in a DNA sequence and subsequent cluster formation algorithm based on the application of electronic filter* | https://doi.org/10.1038/s41598-021-93154-3                                                                  |
