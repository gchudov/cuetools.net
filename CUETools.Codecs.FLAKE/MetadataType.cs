namespace CUETools.Codecs.Flake
{
    public enum MetadataType
    {
        /// <summary>
        /// <A HREF="../format.html#metadata_block_streaminfo">STREAMINFO</A> block
        /// </summary>
        StreamInfo = 0,

        /// <summary>
        /// <A HREF="../format.html#metadata_block_padding">PADDING</A> block
        /// </summary>
        Padding = 1,

        /// <summary>
        /// <A HREF="../format.html#metadata_block_application">APPLICATION</A> block 
        /// </summary>
        Application = 2,

        /// <summary>
        /// <A HREF="../format.html#metadata_block_seektable">SEEKTABLE</A> block
        /// </summary>
        Seektable = 3,

        /// <summary>
        /// <A HREF="../format.html#metadata_block_vorbis_comment">VORBISCOMMENT</A> block (a.k.a. FLAC tags)
        /// </summary>
        VorbisComment = 4,

        /// <summary>
        /// <A HREF="../format.html#metadata_block_cuesheet">CUESHEET</A> block
        /// </summary>
        CUESheet = 5,

        /// <summary>
        /// <A HREF="../format.html#metadata_block_picture">PICTURE</A> block
        /// </summary>
        Picture = 6,

        /// <summary>
        /// marker to denote beginning of undefined type range; this number will increase as new metadata types are added
        /// </summary>
        Undefined = 7
    }
}
