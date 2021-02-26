using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.Logging;
using SongChordsRecognizer.AudioSource;
using SongChordsRecognizer.ErrorMessages;
using SongChordsRecognizer.FourierTransform;
using SongChordsRecognizer.Graphs;
using SongChordsRecognizer.MusicFeatures;
using SongChordsRecognizer.Parsers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using WebSongChordsRecognizer.Models;

namespace WebSongChordsRecognizer.Controllers
{
    public class RecognizerController : Controller
    {
        private readonly ILogger<RecognizerController> _logger;

        public RecognizerController(ILogger<RecognizerController> logger)
        {
            _logger = logger;
        }


        public IActionResult Index()
        {
            return View();
        }

        
        public IActionResult VisualizeChordSequence(IFormFile audio, String windowArg, String filtrationArg, int sampleLengthLevel, int bpm)
        {
            IWindow window;
            ISpectrogramFiltration filtration;

            // Handle exceptions on input
            try
            {
                if (audio == null)
                    throw new Exception(ErrorMessages.RecognizerController_MissingAudio);
                if (!(bpm >= 5 && bpm <= 350))
                    throw new Exception(ErrorMessages.RecognizerController_InvalidSampleLengthLevel);
                if (!(sampleLengthLevel >= 10 && sampleLengthLevel <= 18))
                    throw new Exception(ErrorMessages.RecognizerController_InvalidSampleLengthLevel);

                window = InputArgsParser.ParseSTFTWindow(windowArg);
                filtration = InputArgsParser.ParseFiltration(filtrationArg);
            }
            catch (Exception e)
            {
                return RedirectToAction("IncorrectInputFormat", new { message = e.Message });
            }

            // Create model, process audio, generate chord sequence
            RecognizerModel model = new RecognizerModel();
            model.ProcessAudio(audio, window, filtration, sampleLengthLevel, bpm);

            return View(model);
        }

        public IActionResult About()
        {
            return View();
        }


        public IActionResult IncorrectInputFormat(String message)
        {
            return View(new ErrorMessageModel { Message = message });
        }



        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
