# SimpleTrack C++ Port – Testing Plan

To validate parity between the original Python implementation and the new C++ code, adopt the following workflow:

1. **Deterministic Fixtures**
   - Capture representative `FrameData` snapshots (detections, ego poses, aux info) from the Python pipeline and serialize them (e.g., JSON or binary).
   - Reuse the same fixtures in C++ unit tests to guarantee identical inputs.

2. **Unit Tests**
   - Use a C++ testing framework (GoogleTest or Catch2) to cover:
     - `BBox` conversions and corner generation.
     - Geometry helpers (`iou3d`, `giou3d`, Mahalanobis distance).
     - Association layer (`associate_dets_to_tracks` results vs. Python reference).
     - Kalman filter prediction/update invariants.
     - `HitManager` state transitions.
   - Compare numeric outputs against tolerances derived from Python results.

3. **Integration Tests**
   - Run `MOTModel::frame_mot` across recorded fixture sequences.
   - Cross-check track IDs, states, and bounding boxes against Python reference logs.
   - Ensure the C++ build passes with `-O2` and `-O3` to mirror deployment settings.

4. **Continuous Regression**
   - Integrate tests into CTest via the existing `CMakeLists.txt`.
   - Add GitHub Actions or Jenkins jobs targeting x86_64 and NVIDIA Orin (cross-compilation or on-device CI).

Following this plan provides incremental confidence that the C++ port faithfully mirrors the Python behaviour before deeper optimization work on NVIDIA hardware. 

