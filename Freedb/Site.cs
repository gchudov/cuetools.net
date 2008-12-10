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
	/// Summary description for Site.
	/// </summary>
	public class Site
	{
		private string m_SiteAddress;
		private string m_Protocol;
		private string m_AdditionalAddressInfo;
		private string m_Port;
		private string m_Latitude;
		private string m_Longitude;
		private string m_Description;


		public class PROTOCOLS
		{
			public const string HTTP = "http";
			public const string CDDBP = "cddbp";
			public const string ALL = "all";
		}

		
		/// <summary>
		/// Property AdditionalAddressInfo (string)
		/// Any additional addressing information needed to access the server.
		/// For example, for HTTP protocol servers, this would be the path to the CCDB server CGI script.
		/// This field will be "-" if no additional addressing information is needed.
		/// </summary>
		public string AdditionalAddressInfo
		{
			get
			{
				return this.m_AdditionalAddressInfo;
			}
			set
			{
				this.m_AdditionalAddressInfo = value;
			}
		}
		

		#region Public Properties
		
		/// <summary>
		/// Property Site (string) - Internet address of the remote site 
		/// </summary>
		public string SiteAddress
		{
			get
			{
				return this.m_SiteAddress;
			}
			set
			{
				this.m_SiteAddress = value;
			}
		}

		/// <summary>
		/// Property Protocol (string)
		/// The transfer protocol used to access the site
		/// </summary>
		public string Protocol
		{
			get
			{
				return this.m_Protocol;
			}
			set
			{
				this.m_Protocol = value;
			}
		}

		/// <summary>
		/// Property Port (string)- The port at which the server resides on that site.
		/// </summary>
		public string Port
		{
			get
			{
				return this.m_Port;
			}
			set
			{
				this.m_Port = value;
			}
		}
		


		/// <summary>
		/// Property Description (string)
		/// A short description of the geographical location of the site.
		/// </summary>
		public string Description
		{
			get
			{
				return this.m_Description;
			}
			set
			{
				this.m_Description = value;
			}
		}
		
		
		/// <summary>
		/// Property Latitude (string)
		/// The latitude of the server site. The format is as follows:
		/// CDDD.MM
		/// Where "C" is the compass direction (N, S), "DDD" is the
		/// degrees, and "MM" is the minutes.
		/// </summary>
		public string Latitude
		{
			get
			{
				return this.m_Latitude;
			}
			set
			{
				this.m_Latitude = value;
			}
		}
		
		/// <summary>
		/// Property Longitude (string)
		/// The longitude of the server site. Format is as above, except
		/// the compass direction must be one of (E, W).
		/// </summary>
		public string Longitude
		{
			get
			{
				return this.m_Longitude;
			}
			set
			{
				this.m_Longitude = value;
			}
		}
		

		#endregion


		public Site(string siteFromCDDB)
		{
			if (!Parse(siteFromCDDB))
			{
				throw new Exception("Unable to Parse Site. Input: " + siteFromCDDB);
			}
		}


		/// <summary>
		/// Builds a site from an address, protocol and addition info
		/// </summary>
		/// <param name="siteAddress"></param>
		/// <param name="protocol"></param>
		/// <param name="additionAddressInfo"></param>
		public Site(string siteAddress, string protocol, string additionAddressInfo)
		{
			m_SiteAddress = siteAddress;
			m_Protocol = protocol;
			m_AdditionalAddressInfo = additionAddressInfo;
		}



		public bool Parse(string siteAsString)
		{
			siteAsString.Trim();
			string [] values = siteAsString.Split(' ');
			if (values.Length <5)
				return false;
			m_SiteAddress = values[0];
			this.m_Protocol = values[1];
			m_Port = values[2];
			if (values[3].Trim() != "-")
				m_AdditionalAddressInfo = values[3];
			m_Latitude = values[4];
			m_Longitude = values[5];
			
			// description could be split over many because it could have spaces
			for (int i= 6; i < values.Length;i++)
			{
				m_Description += values[i];
				m_Description += " ";

			}
			m_Description.Trim();
			return true;
		}

		public string GetUrl()
		{

			if (this.m_Protocol == Site.PROTOCOLS.HTTP)
				return "http://" + this.m_SiteAddress + this.m_AdditionalAddressInfo;
			else
				return this.m_SiteAddress;
		}

		public override string ToString()
		{
			return m_SiteAddress + ", " + this.m_Description;
		}



	}
}
