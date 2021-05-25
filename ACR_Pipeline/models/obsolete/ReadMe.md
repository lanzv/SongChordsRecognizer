# Obsolete ACR Models

- basic_mlp.model
  - **MLP_scalered** model
  - *sample_rate = 44100*
  - *hop_length = 22050*
  - *window_size = 5*
  - *skip_coef = 1*
  - *spectrogram = log_mel_spec*
  - original keys


- C_transposed_mlp.model
  - **MLP_scalered** model
  - *sample_rate = 44100*
  - *hop_length = 22050*
  - *window_size = 5*
  - *skip_coef = 1*
  - *spectrogram = log_mel_spec*
  - all songs transposed to C major (ionian)


- original_mlp.model
  - **MLP_scalered** model
  - *sample_rate = 44100*
  - *hop_length = 1024*
  - *window_size = 5*
  - *skip_coef = 22*
  - *spectrogram = log_mel_spec*
  - original keys


- transposed_mlp.model
  - **MLP_scalered** model
  - *sample_rate = 44100*
  - *hop_length = 1024*
  - *window_size = 5*
  - *skip_coef = 22*
  - *spectrogram = log_mel_spec*
  - all songs transposed to C major (ionian)

- old_original_crnn.h5
  - **CRNN_1** model
  - *sample_rate = 22050*
  - *hop_length = 512*
  - *n_frames = 1000*
  - *spectrogram = cqt_spectrogram*
  - original keys in 23 long sequences

- old_transposed_crnn.h5
  - **CRNN_1** model
  - *sample_rate = 22050*
  - *hop_length = 512*
  - *n_frames = 1000*
  - *spectrogram = cqt_spectrogram*
  - all songs transposed to C major (ionian) in 23s long sequences