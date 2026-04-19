# References and Bibliography

Curated list of the academic work that informs this project. Each entry carries a one- to two-line annotation explaining what the paper contributes and why it matters for dual photography.

---

## Foundational Papers

1. **Sen, P., Chen, B., Garg, G., Marschner, S., Horowitz, M., Levoy, M., & Lensch, H. P. A.** (2005). *Dual Photography*. ACM Transactions on Graphics (SIGGRAPH), 24(3), 745-755.
   - The foundational paper. Introduced the concept of dual photography and demonstrated that transposing the measured light transport matrix **T** synthesizes the scene from the projector's viewpoint, by exploiting Helmholtz reciprocity. Basis for this whole repo.

2. **Helmholtz, H. von** (1856). *Handbuch der physiologischen Optik*. Leopold Voss, Leipzig.
   - Original statement of Helmholtz reciprocity — the BRDF is symmetric under swap of incident and exitant directions. The physical principle that makes `dual = T^T · c` physically meaningful.

3. **Debevec, P., Hawkins, T., Tchou, C., Duiker, H.-P., Sarokin, W., & Sagar, M.** (2000). *Acquiring the Reflectance Field of a Human Face*. Proceedings of SIGGRAPH 2000, 145-156.
   - Pioneered measurement of full reflectance fields under controlled illumination. A reflectance field is a superset of the transport matrix, encoding appearance under arbitrary illumination and viewpoint. Direct precursor to transport-matrix imaging.

---

## Acquisition Strategies

4. **Sen, P., & Darabi, S.** (2009). *Compressive Dual Photography*. Computer Graphics Forum (Eurographics), 28(2), 609-618.
   - Applied compressed sensing to dual photography. Reduced acquisition from millions of patterns to hundreds using random Bernoulli illumination and L1-norm reconstruction. Directly motivates this repo's Bernoulli pattern mode.

5. **Peers, P., Mahajan, D. K., Lamond, B., Ghosh, A., Matusik, W., Ramamoorthi, R., & Debevec, P.** (2009). *Compressive Light Transport Sensing*. ACM Transactions on Graphics, 28(1), Article 3.
   - Generalized compressive sensing to the full light transport matrix. Reduced required measurements from O(n^2) to O(n log n) using random multiplexed illumination while preserving reconstruction quality.

6. **Hua, B. S., Sato, I., & Low, K. L.** (2013). *Direct and Progressive Reconstruction of Dual Photography Images*. IEEE International Conference on Image Processing (ICIP), 1860-1864.
   - Progressive reconstruction that produces usable dual images with far fewer measurements. Reduces acquisition from hours to minutes and informs this repo's incremental-rendering strategy.

7. **O'Toole, M., Raskar, R., & Kutulakos, K. N.** (2012). *Temporal Frequency Probing for 5D Transient Light Transport*. ACM Transactions on Graphics (SIGGRAPH), 31(4), Article 87.
   - Extended light transport analysis to the temporal domain. Separates direct and global illumination components and measures light-in-flight. Relevant for understanding multi-bounce contributions that this repo's single-bounce simulator deliberately ignores.

---

## Adjacent Techniques

8. **Ng, R.** (2005). *Fourier Slice Photography*. ACM Transactions on Graphics (SIGGRAPH), 24(3), 735-744.
   - Theoretical framework for 4D light-field photography and refocusing via Fourier-domain slicing. Led to the Lytro camera. Complementary to dual photography: both capture aspects of the 4D light field, dual photography through T and light fields through plenoptic sampling.

9. **Nayar, S. K., Krishnan, G., Grossberg, M. D., & Raskar, R.** (2006). *Fast Separation of Direct and Global Components of a Scene Using High Frequency Illumination*. ACM Transactions on Graphics (SIGGRAPH), 25(3), 935-944.
   - High-frequency pattern projection to separate direct and global illumination in one pass. Conceptually close to dual photography in using patterned illumination as a measurement probe.

10. **Zhang, S.** (2018). *High-speed 3D shape measurement with structured light methods: A review*. Optics and Lasers in Engineering, 106, 119-131.
    - Comprehensive review of Gray code and phase-shifting pattern strategies for projector-camera correspondence. Directly informs `src/core/calibration.py` and the Gray-code pattern generator.

---

## Neural / Learning-Based Light Transport (2019-)

11. **Kang, K., Xie, C., He, C., Yi, M., Gu, M., Chen, Z., Zhou, K., & Wu, H.** (2019). *Learning Efficient Illumination Multiplexing for Joint Capture of Reflectance and Shape*. ACM Transactions on Graphics (SIGGRAPH Asia), 38(6), Article 165.
    - Learns the optimal multiplexed illumination patterns end-to-end, outperforming hand-crafted Hadamard/Bernoulli designs. A promising direction for replacing this repo's fixed pattern set with a learned one.

12. **Maeda, T., Wang, G., Raskar, R., & Kadambi, A.** (2020). *Thermal non-line-of-sight imaging*. IEEE International Conference on Computational Photography (ICCP).
    - Demonstrates transport-matrix-style inference beyond the visible spectrum (thermal NLOS). Motivation for this repo's spectral (multi-wavelength) extension.

13. **Kaya, B., Kumar, S., Sarno, F., Ferrari, V., & Van Gool, L.** (2021). *Neural Radiance Fields Approach to Deep Multi-View Photometric Stereo*. IEEE Winter Conference on Applications of Computer Vision (WACV).
    - Combines neural radiance fields with multi-view photometric stereo. Illustrates how learned implicit scene representations can substitute for explicit transport-matrix storage — an area the repo may explore next.

14. **Gkioulekas, I., Walter, B., Adelson, E. H., Bala, K., & Zickler, T.** (2015). *On the appearance of translucent edges*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 5528-5536.
    - Models how translucent materials violate the diffuse-BRDF assumption used in this repo's Lambertian simulator. Flags a limitation of the single-bounce forward model.

15. **Duran, V., Clemente, P., Fernández-Alonso, M., Tajahuerce, E., & Lancis, J.** (2012). *Single-pixel polarimetric imaging*. Optics Letters, 37(5), 824-826.
    - Early single-pixel polarimetric imaging using patterned illumination and a single photodetector — conceptually dual to dual photography. Relevant to anyone considering hardware-minimal acquisition strategies.

---

## Software and Libraries Used

- **NumPy** — matrix operations, pseudoinverse, dense SVD via LAPACK
- **SciPy** — sparse matrix support, extended SVD options
- **OpenCV** — webcam capture (YUY2 → RGB), image display
- **Dash + Plotly** — interactive web visualization and reactive callbacks
- **Pillow** — PNG encoding for web display
- **scikit-image** — auxiliary image-processing utilities
- **pytest** — 165-test suite covering core, simulation, API, frontend
