#region COPYRIGHT (c) 2004 by Brian Weeres
/* Copyright (c) 2004 by Brian Weeres
 * 
 * Email: bweeres@protegra.com; bweeres@hotmail.com
 * 
 * Permission to use, copy, modify, and distribute this software for any
 * purpose is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * If you modify it then please indicate so. 
 *
 * The software is provided "AS IS" and there are no warranties or implied warranties.
 * In no event shall Brian Weeres and/or Protegra Technology Group be liable for any special, 
 * direct, indirect, or consequential damages or any damages whatsoever resulting for any reason 
 * out of the use or performance of this software
 * 
 */
#endregion
using System;

namespace Freedb
{
	/// <summary>
	/// Summary description for Track.
	/// </summary>
	public class Track
	{

		private string m_Title;
		private string m_ExtendedData;
		
		#region Public Properties
		/// <summary>
		/// Property ExtendedData (string)
		/// </summary>
		public string ExtendedData
		{
			get
			{
				return this.m_ExtendedData;
			}
			set
			{
				this.m_ExtendedData = value;
			}
		}
		
		/// <summary>
		/// Property Title (string)
		/// </summary>
		public string Title
		{
			get
			{
				return this.m_Title;
			}
			set
			{
				this.m_Title = value;
			}
		}
		#endregion





		/// <summary>
		/// Create an instance of a Track 
		/// </summary>
		/// <param name="title"></param>
		public Track()
		{
		}


		/// <summary>
		/// Create an instance of a Track passing in a title
		/// </summary>
		/// <param name="title"></param>
		public Track(string title)
		{
			m_Title = title;
		}

		/// <summary>
		/// Create an instance of a Track passing in a title and extended data
		/// </summary>
		/// <param name="title"></param>
		public Track(string title, string extendedData)
		{
			m_Title = title;
			m_ExtendedData = extendedData;
		}
	
	
	
	}
}
