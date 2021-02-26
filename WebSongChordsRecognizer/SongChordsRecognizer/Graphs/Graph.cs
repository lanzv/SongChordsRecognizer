namespace SongChordsRecognizer.Graphs
{
    /// <summary>
    /// Interface for Graphs like Spectrogram or Chromagram,
    /// that could be printed and which have data in format
    /// double Data[][] where first index corresponds to specific
    /// time frames, second index corresponds to tone pitches.
    /// </summary>
    public interface IGraph
    {
        /// <summary>
        /// Print and visualize data of graph with specific printer type.
        /// </summary>
        /// <param name="printer">IGraphPrinter printer that says how we want to visualize data.</param>
        /// <param name="startingSample">Starting index of time frame samples we want to print.</param>
        /// <param name="length">Number of time frame samples we want to print.</param>
        public void Print(IGraphPrinter printer, int startingSample, int length);
        /// <summary>
        /// GetData creates the copy of Graph Data and returns it as a result.
        /// </summary>
        /// <returns>Graph Data as a new copy.</returns>
        public double[][] GetData();
    }
}