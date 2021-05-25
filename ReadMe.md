# Song Chords Recognizer    

1) Have you ever heard a **song** on YouTube or the radio
that you would like to **play on guitar and sing** with your friends? 

2) Is there some **song** you like and you want to play it **for yourself** or **improvise** on it **with your music friends**?

3) Are you trying to create **sheet music** for a specific **song** for your **band**?

**CHORDS and HARMONY analysis is a very good start!!** - And that is exactly what this application offers.



## ASP.NET Application

Song Chords Recognizer is a [Web Application coded in C#](./WebSongChordsRecognizer/ReadMe.md), where you have two models you can pick to process the audio that will return the chord sequence of the song.

 1. [Predictors](./ACR_Pipeline/ReadMe.md) based on Deep Learning coded in Python.
 2. [Template Voter](./WebSongChordsRecognizer/SongChordsRecognizer/ReadMe.md) based on the simple chord template voting coded in .NET .



## ACR Research

Part of the project is also the [Automatic Chord Recognition task RESEARCH](./ACR_Training/ReadMe.md).
The approach is to use CRNN models with transpose preprocessing and also vote the most frequent chord for each beat duration.
