# Changelog
## 1.0.1 -- 2023-01-06
### Changed
* Face selection is now chunked, to improve performance when more than 1.000 images are processed.

### Fixed
* Fixed a bug where `face_selection_overrides` could not be found in Windows because global variables do not work well
  with concurrency.
* Fixed exception due to incorrect usage of drawing function when multiple faces are detected in a single image.
