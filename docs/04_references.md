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

## Seminal Works

10. **Sen, P., Chen, B., Garg, G., Marschner, S., Horowitz, M., Levoy, M., & Lensch, H. P. A.** (2005). *Dual Photography*. ACM Transactions on Graphics (Proceedings of SIGGRAPH), 24(3), 745-755.
    - The foundational paper that introduced dual photography. Demonstrated that transposing the measured light transport matrix synthesizes the scene from the projector's viewpoint, exploiting Helmholtz reciprocity.

11. **Debevec, P., Hawkins, T., Tchou, C., Duiker, H.-P., Sarokin, W., & Sagar, M.** (2000). *Acquiring the Reflectance Field of a Human Face*. ACM Transactions on Graphics (Proceedings of SIGGRAPH), 331-340.
    - Pioneered the measurement of complete reflectance fields using controlled illumination. The reflectance field is a superset of the transport matrix, encoding appearance under arbitrary illumination and viewpoint.

12. **Ng, R.** (2005). *Fourier Slice Photography*. ACM Transactions on Graphics (Proceedings of SIGGRAPH), 24(3), 735-744. (Based on Stanford PhD thesis: *Digital Light Field Photography*, 2006.)
    - Introduced the theoretical framework for light field photography and refocusing via Fourier-domain slicing. Led to the Lytro camera. Complementary to dual photography in capturing 4D light transport.

13. **Peers, P., Mahajan, D. K., Lamond, B., Ghosh, A., Matusik, W., Ramamoorthi, R., & Debevec, P.** (2009). *Compressive Light Transport Sensing*. ACM Transactions on Graphics, 28(1), Article 3.
    - Applied compressive sensing to light transport acquisition, reducing the number of required measurements from O(n^2) to O(n log n) while preserving reconstruction quality. Uses random multiplexed illumination patterns.

14. **O'Toole, M., Raskar, R., & Kutulakos, K. N.** (2012). *Temporal Frequency Probing for 5D Transient Light Transport*. ACM Transactions on Graphics (Proceedings of SIGGRAPH), 31(4), Article 87.
    - Extended light transport analysis to the temporal domain, enabling separation of direct and global illumination components and measurement of light-in-flight. Relevant to understanding multi-bounce contributions in transport matrices.

## Software and Libraries Used

- **NumPy**: Matrix operations, pseudoinverse computation, SVD
- **SciPy**: Extended SVD options, sparse matrix support
- **OpenCV**: Camera capture, image display, color conversion
- **Dash + Plotly**: Web-based interactive visualization
- **Pillow**: Image encoding for web display
- **scikit-image**: Image processing utilities
