using System;
using System.Collections.Generic;
using System.IO;

namespace SongChordsRecognizer.Graphs
{
    /// <summary>
    /// Interface for all printing functions of IGraph. 
    /// Each of printers should contain function 'Print' that 
    /// print IGraph somewhere and somehow.
    /// </summary>
    public interface IGraphPrinter
    {
        /// <summary>
        /// An algorithm that prints graph.
        /// </summary>
        /// <param name="data">Array of data samples we want to print.</param>
        /// <param name="startingSample">Starting index of data samples we want to print.</param>
        /// <param name="length">Number of data samples we want to print.</param>
        /// <param name="indicesToPrint">List of data indices with their descriptions that supposed to be printed for each data sample.</param>
        /// <param name="sampleLength">Time duration of one data sample in seconds.</param>
        public void Print(double[][] data, int startingSample, int length, List<(int Index, String Description)> indicesToPrint, double sampleLength);
    }



    /// <summary>
    /// Very simple ASCII printer that prints graph from 'data' to the file 'graph.txt'. 
    /// </summary>
    public class PrintGraphToTextFile : IGraphPrinter
    {
        private int lengthOfGraph = 20;
        string file = "graph.txt";

        public void Print(double[][] data, int startingSample, int length, List<(int Index, String Description)> indicesToPrint, double sampleLength)
        {
            if (data.Length == 0) throw new Exception(ErrorMessages.ErrorMessages.GraphPrinters_NoneDataToPrint);

            String[] lines = new String[indicesToPrint.Count + 1];
            lines[0] = "One sample ~ " + sampleLength + "s";

            // create graph description
            for (int i = 1; i < lines.Length; i++) 
            {
                String field = string.Format("{0:N2}", indicesToPrint[i - 1].Description);
                lines[i] = String.Format("|{0,-30}|", field) + " | ";
            }

            // iterate over 'length' data samples with offset 'startingSample'
            for (int i = 0; i < length && startingSample + i < data.Length; i++)
            {
                // add new graph to lines 
                double max = 0;
                for(int j = 1; j < lines.Length; j++)
                {
                    if (max < Math.Abs(data[startingSample+i][indicesToPrint[j - 1].Index])) max = Math.Abs(data[startingSample + i][indicesToPrint[j - 1].Index]);
                }
                for(int j = 1; j < lines.Length; j++)
                {
                    int count = (int)(Math.Abs(lengthOfGraph * (data[startingSample + i][indicesToPrint[j - 1].Index] / max)));
                    
                    lines[j] += new string('+', count);
                    lines[j] += new string(' ', lengthOfGraph - count);
                    lines[j] += " | ";
                }
            }
            PrintToFile(lines);
        }



        private void PrintToFile(String[] lines)
        {
            if (!File.Exists(file))
            {
                File.Create(file).Dispose();
            }
            else if (File.Exists(file))
            {
                using (TextWriter tw = new StreamWriter(file))
                {
                    tw.Flush();
                }
            }
            using (TextWriter tw = new StreamWriter(file))
            {
                for (int i = 0; i < lines.Length; i++)
                {
                    tw.WriteLine(lines[i]);
                }
            }
        }
        


        public void SetFileName(String file)
        {
            this.file = file;
        }
    }
}