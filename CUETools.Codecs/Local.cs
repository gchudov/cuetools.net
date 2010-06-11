using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs
{
	/// <summary>
	/// Localized description attribute
	/// </summary>
	[AttributeUsage(AttributeTargets.All, AllowMultiple = false, Inherited = true)]
	public class SRDescriptionAttribute : DescriptionAttribute
	{
		/// <summary>
		/// Construct the description attribute
		/// </summary>
		/// <param name="text"></param>
		public SRDescriptionAttribute(Type SR, string text)
			: base(text)
		{
			this.localized = false;
			this.SR = SR;
		}

		/// <summary>
		/// Override the return of the description text to localize the text
		/// </summary>
		public override string Description
		{
			get
			{
				if (!localized)
				{
					localized = true;
					this.DescriptionValue = SR.InvokeMember(
						 this.DescriptionValue,
						 System.Reflection.BindingFlags.GetProperty | System.Reflection.BindingFlags.Static |
						 System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic,
						 null,
						 null,
						 new object[] { }) as string;
				}

				return base.Description;
			}
		}

		/// <summary>
		/// Store a flag indicating whether this has been localized
		/// </summary>
		private bool localized;

		/// <summary>
		/// Resource manager to use;
		/// </summary>
		private Type SR;
	}
}
