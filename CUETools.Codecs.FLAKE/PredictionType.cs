namespace CUETools.Codecs.FLAKE
{
    /// <summary>
    /// Type of linear prediction
    /// </summary>
    public enum PredictionType
    {
        /// <summary>
        /// Verbatim
        /// </summary>
        None = 0,
        /// <summary>
        /// Fixed prediction only
        /// </summary>
        Fixed = 1,
        /// <summary>
        /// Levinson-Durbin recursion
        /// </summary>
        Levinson = 2,
        /// <summary>
        /// Exhaustive search
        /// </summary>
        Search = 3
    }
}
