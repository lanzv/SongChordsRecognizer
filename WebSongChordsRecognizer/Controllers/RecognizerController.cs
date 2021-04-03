using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using SongChordsRecognizer.ErrorMessages;
using SongChordsRecognizer.FourierTransform;
using SongChordsRecognizer.Graphs;
using SongChordsRecognizer.Logger;
using SongChordsRecognizer.Parsers;
using System;
using System.Diagnostics;
using WebSongChordsRecognizer.Models;
using WebSongChordsRecognizer.Service;

namespace WebSongChordsRecognizer.Controllers
{
    /// <summary>
    /// ASP.NET Controller serving the SongChordRecognizer application.
    /// </summary>
    public class RecognizerController : Controller
    {
        #region Fields

        /// <summary>
        /// StatisticalModel service class that will process the audio with the Statistical Model.
        /// </summary>
        private readonly StatisticalModel statisticalModel;

        /// <summary>
        /// TemplateVoter service class that will process the audio with the Template Voter Model.
        /// </summary>
        private readonly TemplateVoter templateVoter;

        /// <summary>
        /// Logger of the RecognizerController class.
        /// </summary>
        private readonly ILogger _logger = ApplicationLogging.CreateLogger<RecognizerController>();



        #endregion


        #region Initialization

        public RecognizerController()
        {
            statisticalModel = new StatisticalModel();
            templateVoter = new TemplateVoter();
        }



        #endregion


        #region Endpoints

        /// <summary>
        /// The default page. 
        /// There is a form to upload and process any WAV audio.
        /// </summary>
        /// <returns>IActionResult, HTML View of a form.</returns>
        public IActionResult Index()
        {
            return View();
        }



        /// <summary>
        /// The visualization page.
        /// There is a chord sequence that was generated from uploaded audio file.
        /// It can be runned from Recognizer's Index page via upload form.
        /// </summary>
        /// <param name="audio">IFormFile audio file in WAV format.</param>
        /// <returns>IActionResult, HTML View of a chord sequence, or error page.</returns>
        [HttpPost]
        public IActionResult VisualizeStatisticalModel(IFormFile audio)
        {
            StatisticalModelResponse response;
            // Handle exceptions on input
            if (audio == null) { return RedirectToAction("IncorrectInputFormat", new { message = ErrorMessages.RecognizerController_MissingAudio }); }
            else
            {
                try
                {
                    // process audio, generate chord sequence
                    response = statisticalModel.GetChords(audio);
                }
                catch (Exception e)
                {
                    return RedirectToAction("IncorrectInputFormat", new { message = e.Message });
                }
            }

            return View(response);
        }



        /// <summary>
        /// The visualization page.
        /// There is a chord sequence that was generated from uploaded audio file.
        /// It can be runned from Recognizer's Index page via upload form.
        /// </summary>
        /// <param name="audio">IFormFile audio file in WAV format.</param>
        /// <param name="windowArg">String argument for one of the provided convolutional windows.</param>
        /// <param name="filtrationArg">String argument for one of the provided spectrogram filtrations.</param>
        /// <param name="sampleLengthLevel">Int argument of logarithm of fourier transform length.</param>
        /// <param name="bpm">Int argument of beats per minute</param>
        /// <returns>IActionResult, HTML View of a chord sequence, or error page.</returns>
        [HttpPost]
        public IActionResult VisualizeTemplateVoter(IFormFile audio, String windowArg, String filtrationArg, int sampleLengthLevel, int bpm)
        {
            IWindow window = InputArgsParser.ParseSTFTWindow(windowArg);
            ISpectrogramFiltration filtration = InputArgsParser.ParseFiltration(filtrationArg);
            TemplateVoterResponse response;

            // Handle exceptions on input
            if (audio == null) { return RedirectToAction("IncorrectInputFormat", new { message = ErrorMessages.RecognizerController_MissingAudio }); }
            else if (!(bpm >= 5 && bpm <= 350)) { return RedirectToAction("IncorrectInputFormat", new { message = ErrorMessages.RecognizerController_InvalidSampleLengthLevel }); }
            else if (!(sampleLengthLevel >= 10 && sampleLengthLevel <= 18)) { return RedirectToAction("IncorrectInputFormat", new { message = ErrorMessages.RecognizerController_InvalidSampleLengthLevel }); }
            else if (window == null) { return RedirectToAction("IncorrectInputFormat", new { message = ErrorMessages.Program_NotKnownSTFTWindowType }); }
            else if (filtration == null) { return RedirectToAction("IncorrectInputFormat", new { message = ErrorMessages.Program_NotKnownFiltrationType }); }
            else
            {
                try
                {
                    // process audio, generate chord sequence
                    response = templateVoter.GetChords(audio, window, filtration, sampleLengthLevel, bpm);
                }
                catch (Exception e)
                {
                    return RedirectToAction("IncorrectInputFormat", new { message = e.Message });
                }
            }

            return View(response);
        }



        /// <summary>
        /// The error page.
        /// When error has occured, this page will be showed with appropriate error message.
        /// </summary>
        /// <param name="message">Error message to print.</param>
        /// <returns>IActionResult, HTML View of error message.</returns>
        [HttpGet]
        public IActionResult IncorrectInputFormat(String message)
        {
            return View(new ErrorMessageModel { Message = message });
        }



        /// <summary>
        /// The About Project page.
        /// Informations about the Song Chords Recognizer project.
        /// </summary>
        /// <returns>IActionResult, HTML View with info.</returns>
        [HttpGet]
        public IActionResult About()
        {
            return View();
        }



        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }



        #endregion
    }
}
