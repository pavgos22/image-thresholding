# Image Thresholding Methods in C++ with OpenCV

This project demonstrates multiple grayscale thresholding techniques implemented in C++ using OpenCV. 
It covers basic and advanced methods used in image preprocessing and segmentation.

### 1. Grayscale conversion  
Each pixel is converted using:  
`Y = 0.299·R + 0.587·G + 0.114·B`

### 2. Basic thresholding  
Pixels with brightness > `t` become white (`255`); others become black (`0`).

### 3. Pseudo-thresholding  
Pixels above threshold `t` keep original brightness; others are set to `0`.

### 4. Double thresholding  
Pixels in range `[t1, t2]` become white (`255`); others become black (`0`).

### 5. Gradient-based thresholding  
Threshold `t` is computed as:  
`t = Σ(J(x,y)·G(x,y)) / Σ(G(x,y))`  
where `G(x,y) = max(|Gx|, |Gy|)` is the local brightness gradient.

### 6. Iterative thresholding using histogram  
- Histogram is normalized.  
- Pixels are split into two classes by threshold `t`.  
- Means `μ₀` and `μ₁` are computed for pixels below and above `t`.  
- New threshold: `t = (μ₀ + μ₁) / 2`.  
- Repeat until `|tₙ₊₁ - tₙ| < ε`.

---

## 🔧 Compilation & Running

Make sure OpenCV is installed. Then compile and run using:

```bash
g++ main.cpp -o thresholding `pkg-config --cflags --libs opencv4`
```
```bash
./thresholding
```
