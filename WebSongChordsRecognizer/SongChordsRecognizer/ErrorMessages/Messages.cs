namespace SongChordsRecognizer.ErrorMessages
{
    public static class ErrorMessages
    {
        public static string IWindow_NotCorrespondingLength = "\n\nGiven offset and length of window for array 'g' does not correspond to g's length.";
        public static string AudioSource_NotSupportedBitsPerSample = "\n\nThe number of bits per waveform sample that is not indivisible by 8 is not supported.";
        public static string GraphPrinters_NoneDataToPrint = "\n\nThe GraphPrinters tries to print empty Graph with 0 length of Data.";
        public static string GraphSumSamples_InvalidLength = "\n\nThe length in SumSamples function has to be non negative and non zero integer.";
        public static string GraphSumSamples_InvalidOffset= "\n\nThe offset in SumSamples function has to be in range of number of all samples in array.";
        public static string SpectrogramFiltration_OctaveOutOfRange = "\n\nWeightedOctaves tries to change weight of octave that does not exist.";
        public static string ChordGenerator_InvalidIntervalFormat = "\n\nProgram tries to get description of chord with invalid third, fifth or seventh interval format.";
        public static string Program_WrongNumberOfArguments = "\n\nThere was passed invalid number of arguments. The correct format of arguments is \n[path of audio file] [STFT window type] [Spectrogram filtration type] [time length level of one sample] [BPM value]\n";
        public static string Program_NotKnownSTFTWindowType = "\n\nThe second argument is invalid.\nOnly these STFT window types are supported: {Rectangular, Triangular, Parzen, Welch, Nuttall}.\n";
        public static string Program_NotKnownFiltrationType = "\n\nThe third argument is invalid.\nOnly these filtration types are supported: {Identity, AFAM, WO, FNH}.\n";
        public static string Program_InvalidSampleLengthLevel = "\n\nThe fourth argument is invalid.\nThe fourth argument has to be non negative integer.\n";
        public static string Program_InvalidBPM = "\n\nThe fifth argument is invalid.\nThe bpm argument has to be non negative integer.\n";
        public static string RecognizerController_InvalidSampleLengthLevel = "\n\nThe fourth argument is invalid.\nThe fourth argument has to be non negative integer in range of 10 and 18.\n";
        public static string RecognizerController_InvalidBPM = "\n\nThe fifth argument is invalid.\nThe bpm argument has to be non negative integer in range of 20 and 250.\n";
        public static string RecognizerController_MissingAudio = "\n\nThe audio file haven't been uploaded.";
    }
}