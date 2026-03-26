# References and Bibliography

## Foundational Papers

1. **Sen, P., Chen, B., Garg, G., Marschner, S., Horowitz, M., Levoy, M., & Lensch, H. P. A.** (2005). *Dual Photography*. ACM Transactions on Graphics (SIGGRAPH), 24(3), 745-755.
   - Introduced the concept of dual photography and demonstrated that transposing the light transport matrix synthesizes the view from the projector's position.

2. **Sen, P., & Darabi, S.** (2009). *Compressive Dual Photography*. Computer Graphics Forum (Eurographics), 28(2), 609-618.
   - Applied compressed sensing to reduce acquisition from millions of patterns to hundreds, using random Bernoulli illumination and L1-norm reconstruction.

3. **Hua, B. S., Sato, I., & Low, K. L.** (2013). *Direct and Progressive Reconstruction of Dual Photography Images*. IEEE International Conference on Image Processing (ICIP).
   - Proposed progressive reconstruction methods that produce usable dual images with far fewer measurements, reducing computation from hours to minutes.

## Related Techniques

4. **Helmholtz Reciprocity**: The physical principle underlying dual photography. States that the BRDF is symmetric: swapping incident and exitant directions yields the same reflectance. Based on time-reversal symmetry of Maxwell's equations.

5. **Structured Light**: Gray code and sinusoidal pattern projection for establishing projector-camera pixel correspondences. Used in some transport matrix acquisition methods.

6. **Radiosity**: Global illumination algorithm based on form factors between surface patches. Used in this project's inter-reflection simulation.

## Recent Developments (2020-2025)

7. Camera-free 3D dual photography using dual DMDs and single-pixel detectors (Optics Express, 2020).

8. Hierarchical orthogonal codes for faster transport matrix acquisition (SPIE/ICVIP, 2024).

9. Neural implicit representations of light transport, replacing explicit matrix storage with learned functions.

## Software and Libraries Used

- **NumPy**: Matrix operations, pseudoinverse computation, SVD
- **SciPy**: Extended SVD options, sparse matrix support
- **OpenCV**: Camera capture, image display, color conversion
- **Dash + Plotly**: Web-based interactive visualization
- **Pillow**: Image encoding for web display
- **scikit-image**: Image processing utilities
