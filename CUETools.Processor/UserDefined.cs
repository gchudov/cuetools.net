//
// File.cs: Provides tagging and properties support for WavPack files.
//
// Author:
//   Brian Nickel (brian.nickel@gmail.com)
//
// Original Source:
//   wvfile.cpp from libtunepimp
//
// Copyright (C) 2006-2007 Brian Nickel
// Copyright (C) 2006 by Lukáš Lalinský (Original Implementation)
// Copyright (C) 2004 by Allan Sandfeld Jensen (Original Implementation)
//
// This library is free software; you can redistribute it and/or modify
// it  under the terms of the GNU Lesser General Public License version
// 2.1 as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
// USA
//

using System;
using System.Collections.Generic;
using TagLib;
using CUETools.Codecs;

namespace TagLib.UserDefined {
	/// <summary>
	///    This class extends <see cref="TagLib.NonContainer.File" /> to
	///    provide tagging and properties support for user defined format files.
	/// </summary>
	/// <remarks>
	///    A <see cref="TagLib.Ape.Tag" /> will be added automatically to
	///    any file that doesn't contain one. This change does not effect
	///    the file and can be reversed using the following method:
	///    <code>file.RemoveTags (file.TagTypes &amp; ~file.TagTypesOnDisk);</code>
	/// </remarks>
	[SupportedMimeType("taglib/misc", "misc")]
	public class File : TagLib.NonContainer.File
	{
		#region Private Fields

		private CUEToolsTagger tagger;
		
		#endregion
		
		
		
		#region Constructors
		
		/// <summary>
		///    Constructs and initializes a new instance of <see
		///    cref="File" /> for a specified path in the local file
		///    system and specified read style.
		/// </summary>
		/// <param name="path">
		///    A <see cref="string" /> object containing the path of the
		///    file to use in the new instance.
		/// </param>
		/// <param name="propertiesStyle">
		///    A <see cref="ReadStyle" /> value specifying at what level
		///    of accuracy to read the media properties, or <see
		///    cref="ReadStyle.None" /> to ignore the properties.
		/// </param>
		/// <exception cref="ArgumentNullException">
		///    <paramref name="path" /> is <see langword="null" />.
		/// </exception>
		public File (string path, ReadStyle propertiesStyle, CUEToolsTagger _tagger)
			: base (path, propertiesStyle)
		{
			tagger = _tagger;
			// Make sure we have a tag.
			switch (tagger)
			{
                case CUEToolsTagger.APEv2:
					GetTag(TagTypes.Ape, true);
					break;
                case CUEToolsTagger.ID3v2:
					GetTag(TagTypes.Id3v2, true);
					break;
			}
		}
		
		/// <summary>
		///    Constructs and initializes a new instance of <see
		///    cref="File" /> for a specified path in the local file
		///    system with an average read style.
		/// </summary>
		/// <param name="path">
		///    A <see cref="string" /> object containing the path of the
		///    file to use in the new instance.
		/// </param>
		/// <exception cref="ArgumentNullException">
		///    <paramref name="path" /> is <see langword="null" />.
		/// </exception>
        public File(string path, CUEToolsTagger _tagger)
			: base(path)
		{
			tagger = _tagger;
			// Make sure we have a tag.
			switch (tagger)
			{
                case CUEToolsTagger.APEv2:
					GetTag(TagTypes.Ape, true);
					break;
                case CUEToolsTagger.ID3v2:
					GetTag(TagTypes.Id3v2, true);
					break;
			}
		}
		
		/// <summary>
		///    Constructs and initializes a new instance of <see
		///    cref="File" /> for a specified file abstraction and
		///    specified read style.
		/// </summary>
		/// <param name="abstraction">
		///    A <see cref="IFileAbstraction" /> object to use when
		///    reading from and writing to the file.
		/// </param>
		/// <param name="propertiesStyle">
		///    A <see cref="ReadStyle" /> value specifying at what level
		///    of accuracy to read the media properties, or <see
		///    cref="ReadStyle.None" /> to ignore the properties.
		/// </param>
		/// <exception cref="ArgumentNullException">
		///    <paramref name="abstraction" /> is <see langword="null"
		///    />.
		/// </exception>
		public File (File.IFileAbstraction abstraction,
                     ReadStyle propertiesStyle, CUEToolsTagger _tagger)
			: base (abstraction, propertiesStyle)
		{
			tagger = _tagger;
			// Make sure we have a tag.
			switch (tagger)
			{
                case CUEToolsTagger.APEv2:
					GetTag(TagTypes.Ape, true);
					break;
                case CUEToolsTagger.ID3v2:
					GetTag(TagTypes.Id3v2, true);
					break;
			}
		}
		
		/// <summary>
		///    Constructs and initializes a new instance of <see
		///    cref="File" /> for a specified file abstraction with an
		///    average read style.
		/// </summary>
		/// <param name="abstraction">
		///    A <see cref="IFileAbstraction" /> object to use when
		///    reading from and writing to the file.
		/// </param>
		/// <exception cref="ArgumentNullException">
		///    <paramref name="abstraction" /> is <see langword="null"
		///    />.
		/// </exception>
        public File(File.IFileAbstraction abstraction, CUEToolsTagger _tagger)
			: base (abstraction)
		{
			tagger = _tagger;
			// Make sure we have a tag.
			switch (tagger)
			{
                case CUEToolsTagger.APEv2:
					GetTag(TagTypes.Ape, true);
					break;
                case CUEToolsTagger.ID3v2:
					GetTag(TagTypes.Id3v2, true);
					break;
			}
		}
		
		#endregion
		
		
		
		#region Public Methods

        public CUEToolsTagger Tagger
		{
			get
			{
				return tagger;
			}
		}	

		/// <summary>
		///    Gets a tag of a specified type from the current instance,
		///    optionally creating a new tag if possible.
		/// </summary>
		/// <param name="type">
		///    A <see cref="TagLib.TagTypes" /> value indicating the
		///    type of tag to read.
		/// </param>
		/// <param name="create">
		///    A <see cref="bool" /> value specifying whether or not to
		///    try and create the tag if one is not found.
		/// </param>
		/// <returns>
		///    A <see cref="Tag" /> object containing the tag that was
		///    found in or added to the current instance. If no
		///    matching tag was found and none was created, <see
		///    langword="null" /> is returned.
		/// </returns>
		/// <remarks>
		///    If a <see cref="TagLib.Id3v2.Tag" /> is added to the
		///    current instance, it will be placed at the start of the
		///    file. On the other hand, <see cref="TagLib.Id3v1.Tag" />
		///    <see cref="TagLib.Ape.Tag" /> will be added to the end of
		///    the file. All other tag types will be ignored.
		/// </remarks>
		public override TagLib.Tag GetTag (TagTypes type, bool create)
		{
			Tag t = (Tag as TagLib.NonContainer.Tag).GetTag (type);
			
			if (t != null || !create)
				return t;
			
			switch (type)
			{
			case TagTypes.Id3v1:
				return EndTag.AddTag (type, Tag);
			
			case TagTypes.Id3v2:
				return StartTag.AddTag (type, Tag);
			
			case TagTypes.Ape:
				return EndTag.AddTag (type, Tag);
			
			default:
				return null;
			}
		}
		
		#endregion
		
		
		
		#region Protected Methods
		
		/// <summary>
		///    Reads format specific information at the start of the
		///    file.
		/// </summary>
		/// <param name="start">
		///    A <see cref="long" /> value containing the seek position
		///    at which the tags end and the media data begins.
		/// </param>
		/// <param name="propertiesStyle">
		///    A <see cref="ReadStyle" /> value specifying at what level
		///    of accuracy to read the media properties, or <see
		///    cref="ReadStyle.None" /> to ignore the properties.
		/// </param>
		protected override void ReadStart (long start,
		                                   ReadStyle propertiesStyle)
		{
		}
		
		/// <summary>
		///    Reads format specific information at the end of the
		///    file.
		/// </summary>
		/// <param name="end">
		///    A <see cref="long" /> value containing the seek position
		///    at which the media data ends and the tags begin.
		/// </param>
		/// <param name="propertiesStyle">
		///    A <see cref="ReadStyle" /> value specifying at what level
		///    of accuracy to read the media properties, or <see
		///    cref="ReadStyle.None" /> to ignore the properties.
		/// </param>
		protected override void ReadEnd (long end,
		                                 ReadStyle propertiesStyle)
		{
		}
		
		/// <summary>
		///    Reads the audio properties from the file represented by
		///    the current instance.
		/// </summary>
		/// <param name="start">
		///    A <see cref="long" /> value containing the seek position
		///    at which the tags end and the media data begins.
		/// </param>
		/// <param name="end">
		///    A <see cref="long" /> value containing the seek position
		///    at which the media data ends and the tags begin.
		/// </param>
		/// <param name="propertiesStyle">
		///    A <see cref="ReadStyle" /> value specifying at what level
		///    of accuracy to read the media properties, or <see
		///    cref="ReadStyle.None" /> to ignore the properties.
		/// </param>
		/// <returns>
		///    A <see cref="TagLib.Properties" /> object describing the
		///    media properties of the file represented by the current
		///    instance.
		/// </returns>
		protected override Properties ReadProperties (long start,
		                                              long end,
		                                              ReadStyle propertiesStyle)
		{
			return new Properties ();
		}
		
		#endregion
	}

	public static class AdditionalFileTypes
	{
		private static bool inited = false;
        private static CUEToolsCodecsConfig _config;

        public static CUEToolsCodecsConfig Config
		{
			set
			{
				Init();
				_config = value;
			}
		}

		private static TagLib.File UserDefinedResolver(TagLib.File.IFileAbstraction abstraction, string mimetype, TagLib.ReadStyle style)
		{
			foreach (KeyValuePair<string,CUEToolsFormat> fmt in _config.formats)
                if (fmt.Value.tagger != CUEToolsTagger.TagLibSharp && mimetype == "taglib/" + fmt.Key)
					return new File(abstraction, style, fmt.Value.tagger);
			return null;
		}

		static AdditionalFileTypes ()
		{
			Init();
		}

		internal static void Init()
		{
			if (inited)
			    return;
			TagLib.File.AddFileTypeResolver(new TagLib.File.FileTypeResolver(UserDefinedResolver));
			//FileTypes.Register(typeof(TagLib.NonContainer.File));
			inited = true;
		}
	}
}
