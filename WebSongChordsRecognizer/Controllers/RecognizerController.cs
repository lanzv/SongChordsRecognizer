using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.Logging;
using SongChordsRecognizer.AudioSource;
using SongChordsRecognizer.FourierTransform;
using SongChordsRecognizer.Graphs;
using SongChordsRecognizer.MusicFeatures;
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

        
        public IActionResult VisualizeChordSequence(IFormFile audio)
        {
            int sampleLengthLevel = 14;
            IWindow window = new WelchWindow();
            ISpectrogramFiltration filtration = new WeightedOctaves();
            int bpm = 120;


            using (var filestream = new FileStream("output.wav", FileMode.Create, FileAccess.Write))
            {
                audio.CopyTo(filestream);
            }
            RecognizerModel model = new RecognizerModel();
            model.ProcessAudio("output.wav", window, filtration, sampleLengthLevel, bpm);

            return View(model);
        }

        public IActionResult About()
        {
            return View();
        }






        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
