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
using System.Collections.Specialized;
using System.Diagnostics;
using System.Text;

namespace Freedb
{
	/// <summary>
	/// Summary description for CDEntry.
	/// </summary>
	public class CDEntry
	{


		
		#region Private Member Variables
		private string m_Discid;
		private string m_Artist;
		private string m_Title;
		private string m_Year;
		private string m_Genre;
		private TrackCollection m_Tracks = new TrackCollection(); // 0 based - first track is at 0 last track is at numtracks - 1
		private string m_ExtendedData;
		private string m_PlayOrder;
		
		/// <summary>
		/// Property NumberOfTracks (int)
		/// </summary>
		


		
		#endregion

		#region Public Member Variables
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
		
		/// <summary>
		/// Property Year (string)
		/// </summary>
		public string Year
		{
			get
			{
				return this.m_Year;
			}
			set
			{
				this.m_Year = value;
			}
		}

		/// <summary>
		/// Property Genre (string)
		/// </summary>
		public string Genre
		{
			get
			{
				return this.m_Genre;
			}
			set
			{
				this.m_Genre = value;
			}
		}


		/// <summary>
		/// Property Tracks (StringCollection)
		/// </summary>
		public TrackCollection Tracks
		{
			get
			{
				return this.m_Tracks;
			}
			set
			{
				this.m_Tracks = value;
			}
		}

		
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
		/// Property PlayOrder (string)
		/// </summary>
		public string PlayOrder
		{
			get
			{
				return this.m_PlayOrder;
			}
			set
			{
				this.m_PlayOrder = value;
			}
		}

		public int NumberOfTracks
		{
			get
			{
				return m_Tracks.Count;
			}
		}

		#endregion
		
		
		
		public CDEntry(StringCollection data)
		{
			if (!Parse(data))
			{
				throw new Exception("Unable to Parse CDEntry.");
			}
		}


		private bool Parse(StringCollection data)
		{
			int offsetNumber = -1;
			foreach (string line in data)
			{

				// check for comment

				if (line[0] == '#')
				{
					if (offsetNumber == -2)
						continue;
					if (offsetNumber == -1)
					{
						if (line.Substring(1).Trim() != "Track frame offsets:")
							continue;
						offsetNumber = 0;
						continue;
					}
					if (line.Substring(1).Trim() == "" || line.Substring(1).Trim()[0] == 'D')
					{
						offsetNumber = -2;
						continue;
					}
					int offset;
					if (!int.TryParse(line.Substring(1).Trim(), out offset))
					{
						Debug.WriteLine("Failed to parse track FrameOffset: " + line);
						continue;
					}
					//may need to concatenate track info
					while (offsetNumber >= m_Tracks.Count)
						this.m_Tracks.Add(new Track(""));
					m_Tracks[offsetNumber].FrameOffset = offset;
					offsetNumber++;
					continue;
				}

				int index = line.IndexOf('=');
				if (index == -1) // couldn't find equal sign have no clue what the data is
					continue;
				string field = line.Substring(0,index);
				index++; // move it past the equal sign

				switch (field)
				{
					case "DISCID":
					{
						this.m_Discid = line.Substring(index);
						continue;
					}

					case "DTITLE": // artist / title
					{
						this.m_Artist += line.Substring(index);
						continue;
					}

					case "DYEAR":
					{
						this.m_Year = line.Substring(index);
						continue;
					}

					case "DGENRE":
					{
						this.m_Genre += line.Substring(index);
						continue;
					}

					case "EXTD":
					{
						// may be more than one - just concatenate them
						this.m_ExtendedData += line.Substring(index);
						continue;
					}

					case "PLAYORDER":
					{
						this.m_PlayOrder += line.Substring(index);
						continue;
					}

					
					default:

						//get track info or extended track info
						if (field.StartsWith("TTITLE"))
						{
							int trackNumber = -1;
							// Parse could throw an exception
							try
							{
								trackNumber = int.Parse(field.Substring("TTITLE".Length));
							}
							
							catch (Exception ex)
							{
								Debug.WriteLine("Failed to parse track Number. Reason: " + ex.Message);
								continue;
							}

							//may need to concatenate track info
							if (trackNumber < m_Tracks.Count )
								m_Tracks[trackNumber].Title += line.Substring(index);
							else
							{
								Track track = new Track(line.Substring(index));
								this.m_Tracks.Add(track);
							}
							continue;
						}
						else if (field.StartsWith("EXTT"))
						{
							int trackNumber = -1;
							// Parse could throw an exception
							try
							{
								trackNumber = int.Parse(field.Substring("EXTT".Length));
							}
							
							catch (Exception ex)
							{
								Debug.WriteLine("Failed to parse track Number. Reason: " + ex.Message);
								continue;
							}
							
							if (trackNumber < 0 || trackNumber >  m_Tracks.Count -1)
								continue;

							m_Tracks[trackNumber].ExtendedData += line.Substring(index);



						}




						continue;

				} //end of switch

			}


			//split the title and artist from DTITLE;
			// see if we have a slash
			int slash = this.m_Artist.IndexOf(" / ");
			if (slash == -1)
			{
				this.m_Title= m_Artist;
			}
			else
			{
				string titleArtist = m_Artist;
				this.m_Artist = titleArtist.Substring(0,slash);
				slash +=3; // move past " / "
				this.m_Title  = titleArtist.Substring(slash );
			}


			return true;


		}

		public override string ToString()
		{
			StringBuilder builder = new StringBuilder();
			builder.Append("Title: ");
			builder.Append(this.m_Title);
			builder.Append("\n");
			builder.Append("Artist: ");
			builder.Append(this.m_Artist);
			builder.Append("\n");
			builder.Append("Discid: ");
			builder.Append(this.m_Discid);
			builder.Append("\n");
			builder.Append("Genre: ");
			builder.Append(this.m_Genre);
			builder.Append("\n");
			builder.Append("Year: ");
			builder.Append(this.m_Year);
			builder.Append("\n");
			builder.Append("Tracks:");
			foreach (Track track in this.m_Tracks)
			{
				builder.Append("\n");
				builder.Append(track.Title);
			}

			return builder.ToString();

		}

















	}
}
