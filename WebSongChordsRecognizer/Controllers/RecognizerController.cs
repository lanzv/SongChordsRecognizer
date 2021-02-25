using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using SongChordsRecognizer.AudioSource;
using SongChordsRecognizer.FourierTransform;
using SongChordsRecognizer.Graphs;
using SongChordsRecognizer.MusicFeatures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
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

        public IActionResult UploadAudio(IFormFile audio)
        {
            String audioPath = "Let it be 120bpm.wav";
            int sampleLengthLevel = 14;
            IWindow window = new WelchWindow();
            ISpectrogramFiltration filtration = new WeightedOctaves();
            int bpm = 120;

            // Generate chords
            AudioSourceWav wav = new AudioSourceWav(audioPath);

            // SPECTROGRAM
            // - generate
            Spectrogram spectrogram = new Spectrogram(wav, sampleLengthLevel, window);

            // CHROMAGRAM
            // - generate
            Chromagram chromagram = new Chromagram(spectrogram, filtration);


            // CHORD CLASSIFIER
            List<Chord> chords = ChordClassifier.GetChords(chromagram, bpm);




            // ----------------- PRINT CHORDS -----------------
            Console.WriteLine();
            Console.WriteLine(new String('-', 56) + " CHORDS " + new String('-', 56));
            Console.WriteLine();
            for (int i = 0; i < chords.Count; i++)
            {
                Console.Write(chords[i].Description.PadRight(10));
                if ((i + 1) % 12 == 0) Console.WriteLine();
            }
            Console.WriteLine();
            Console.WriteLine(new String('-', 56) + " CHORDS " + new String('-', 56));
            Console.WriteLine();
            // -------------------------------------------------
            return RedirectToAction("Index");
        }

        public IActionResult Privacy()
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
