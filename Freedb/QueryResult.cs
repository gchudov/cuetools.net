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
	/// Summary description for QueryResult.
	/// </summary>
	public class QueryResult
	{
		private string m_ResponseCode;
		private string m_Category;
		private string m_Discid;
		private string m_Artist;
		private string m_Title;
		
		
		#region Public Properties
		/// <summary>
		/// Property ResponseCode (string)
		/// </summary>
		public string ResponseCode
		{
			get
			{
				return this.m_ResponseCode;
			}
			set
			{
				this.m_ResponseCode = value;
			}
		}


		/// <summary>
		/// Property Category (string)
		/// </summary>
		public string Category
		{
			get
			{
				return this.m_Category;
			}
			set
			{
				this.m_Category = value;
			}
		}

		/// <summary>
		/// Property Discid (string)
		/// </summary>
		public string Discid
		{
			get
			{
				return this.m_Discid;
			}
			set
			{
				this.m_Discid = value;
			}
		}

		/// <summary>
		/// Property Artist (string)
		/// </summary>
		public string Artist
		{
			get
			{
				return this.m_Artist;
			}
			set
			{
				this.m_Artist = value;
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

		public QueryResult(string queryResult)
		{
			if (!Parse(queryResult,false))
			{
				throw new Exception("Unable to Parse QueryResult. Input: " + queryResult);
			}

		}

		/// <summary>
		/// The parsing for a queryresult returned as part of a number of matches is slightly different
		/// There is no response code
		/// </summary>
		/// <param name="queryResult"></param>
		/// <param name="multiMatchInput"> true if the result is part of multi-match which means it will not contain a response code</param>
		public QueryResult(string queryResult, bool multiMatchInput)
		{
			if (!Parse(queryResult,multiMatchInput))
			{
				throw new Exception("Unable to Parse QueryResult. Input: " + queryResult);
			}

		}

		/// <summary>
		/// Parse the query result line from the cddb server
		/// </summary>
		/// <param name="queryResult"></param>
		public bool Parse(string queryResult,bool match)
		{

			queryResult.Trim();
			int secondIndex =0;
			
			// get first white space
			int index = queryResult.IndexOf(' ');
			//if we are parsing a matched queryresult there is no responsecode so skip it
			if (!match)
			{
				m_ResponseCode = queryResult.Substring(0,index);
				index++;
				secondIndex = queryResult.IndexOf(' ',index);
			}
			else 
			{
				secondIndex = index;
				index=0;
			}

			m_Category = queryResult.Substring(index,secondIndex-index);
			index = secondIndex;
			index++;
			secondIndex = queryResult.IndexOf(' ',index);
			m_Discid = queryResult.Substring(index,secondIndex-index);
			index = secondIndex;
			index++;
			secondIndex = queryResult.IndexOf('/',index);
			m_Artist = queryResult.Substring(index,secondIndex-index-1); // -1 because there is a space at the end of artist
			index = secondIndex;
			index+=2; //skip past / and space
			m_Title = queryResult.Substring(index); 
			return true;
		}

//		public bool Parse(string queryResult)
//		{
//			queryResult.Trim();
//			string [] values = queryResult.Split(' ');
//			if (values.Length <6)
//				return false;
//			this.m_ResponseCode = values[0];
//			m_Category = values[1];
//			m_Discid = values[2];
//
//			// now we need to look for a slash
//			bool artist = true;
//			for (int i = 3; i < values.Length;i++)
//			{
//				if (values[i] == "/")
//				{
//					artist = false;
//					continue;
//				}
//				if (artist)
//					this.m_Artist += values[i];
//				else
//					this.m_Title += values[i];
//
//			}
//			return true;
//		}

		public override string ToString()
		{
			return this.m_Artist + ", " + this.m_Title;
		}

	}
}
