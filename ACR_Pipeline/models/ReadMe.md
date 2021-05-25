# ACR Models

- original_crnn.h5
  - **CRNN_basic_WithStandardScaler** model, using original_preprocessor.bin as a trained StandardScaler
  - *sample_rate = 22050*
  - *hop_length = 512*
  - *n_frames = 1000*
  - *spectrogram = cqt_spectrogram*
  - original keys in 23 long sequences

- transposed_crnn.h5
  - **CRNN_basic_WithStandardScaler** model, using transposed_preprocessor.bin as a trained StandardScaler
  - *sample_rate = 22050*
  - *hop_length = 512*
  - *n_frames = 1000*
  - *spectrogram = cqt_spectrogram*
  - all songs transposed to C major (ionian) in 23s long sequences