namespace CUETools.Processor
{
    public enum CUEStyle
    {
        /// <summary>
        /// Single file with embedded CUE
        /// </summary>
        SingleFileWithCUE,
        /// <summary>
        /// Single file with external CUE
        /// </summary>
        SingleFile,
        /// <summary>
        /// Gaps prepended file-per-track
        /// </summary>
        GapsPrepended,
        /// <summary>
        /// Gaps appended (noncompliant) file-per-track
        /// </summary>
        GapsAppended,
        /// <summary>
        /// Gaps left out file-per-track
        /// </summary>
        GapsLeftOut
    }
}
