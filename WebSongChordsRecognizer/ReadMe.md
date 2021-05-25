# Song Chords Recognizer - Web ASP.NET Application

## Configuration
Edit ```appsettings.json``` config file.

!! Fill the ```Predictors.PythonPath``` parameter by the path to your python.exe !!

Other parameters are optional.


## Recognizer Controller

#### ***/index***

Default index page where you can upload and process an audio file with any of provided models.

- Method: GET
- Arguments: None

![index_p prtsc](./docs/imgs/index_p.png)

![index_tv prtsc](./docs/imgs/index_tv.png)


#### ***/VisualizePredictors***
Outcome of [Song Chords Recognizer](../ACR_Pipeline/ReadMe.md) model based on Deep Learning with its key and bpm.

- Method: POST
- Arguments: IFormFile audio

![VisualizeStatisticalModel prtsc](./docs/imgs/VisualizePredictors.png)


#### ***/VisualizeTemplateVoter***
Outcome of [Song Chords Recognizer](./SongChordsRecognizer/ReadMe.md) model based on the simple template voting from generated and filtered spectrograms.

- Method: POST
- Arguments: IFormFile audio, String windowArg, String filtrationArg, int sampleLengthLevel, int bpm

![VisualizeTemplateVoter prtsc](./docs/imgs/VisualizeTemplateVoter.png)


#### ***/About***
Basic information about the project with the GitHub link.

- Method: GET
- Arguments: None

![About prtsc](./docs/imgs/About.png)


#### */IncorrectInputFormat*
An error message when some error occures.

- Method: GET
- Arguments: String messages

![IncorrectInputFormat prtsc](./docs/imgs/IncorrectInputFormat.png)

