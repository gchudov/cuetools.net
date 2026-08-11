using System;
using System.Collections.Generic;
using System.Data;
using System.IO;

namespace CUEPlayer {


    public partial class DataSet1 : DataSet {
		private readonly PlaylistDataTable tablePlaylist;

		public DataSet1()
		{
			DataSetName = "DataSet1";
			tablePlaylist = new PlaylistDataTable();
			Tables.Add(tablePlaylist);
		}

		public PlaylistDataTable Playlist
		{
			get { return tablePlaylist; }
		}

		public partial class PlaylistDataTable : DataTable
		{
			private readonly DataColumn columnId;
			private readonly DataColumn columnPath;
			private readonly DataColumn columnArtist;
			private readonly DataColumn columnTitle;
			private readonly DataColumn columnAlbum;
			private readonly DataColumn columnLength;
			private readonly DataColumn columnTrack;

			public PlaylistDataTable()
				: base("Playlist")
			{
				columnId = Columns.Add("id", typeof(int));
				columnId.AutoIncrement = true;
				columnId.AutoIncrementSeed = -1;
				columnId.AutoIncrementStep = -1;
				columnPath = Columns.Add("path", typeof(string));
				columnArtist = Columns.Add("artist", typeof(string));
				columnTitle = Columns.Add("title", typeof(string));
				columnAlbum = Columns.Add("album", typeof(string));
				columnLength = Columns.Add("length", typeof(int));
				columnTrack = Columns.Add("track", typeof(int));
				PrimaryKey = new[] { columnId };
			}

			public PlaylistRow this[int index]
			{
				get { return (PlaylistRow)Rows[index]; }
			}

			public PlaylistRow AddPlaylistRow(string path, string artist, string title, string album, int length, int track)
			{
				PlaylistRow row = (PlaylistRow)NewRow();
				row.path = path;
				row.artist = artist;
				row.title = title;
				row.album = album;
				row.length = length;
				row.track = track;
				Rows.Add(row);
				return row;
			}

			public IEnumerator<PlaylistRow> GetEnumerator()
			{
				foreach (DataRow row in Rows)
					if (row.RowState != DataRowState.Deleted)
						yield return (PlaylistRow)row;
			}

			protected override Type GetRowType()
			{
				return typeof(PlaylistRow);
			}

			protected override DataRow NewRowFromBuilder(DataRowBuilder builder)
			{
				return new PlaylistRow(builder);
			}
		}

		public partial class PlaylistRow : DataRow
		{
			internal PlaylistRow(DataRowBuilder builder)
				: base(builder)
			{
			}

			public int id
			{
				get { return FieldValue<int>("id"); }
				set { this["id"] = value; }
			}

			public string path
			{
				get { return FieldValue<string>("path"); }
				set { this["path"] = value ?? (object)DBNull.Value; }
			}

			public string artist
			{
				get { return FieldValue<string>("artist"); }
				set { this["artist"] = value ?? (object)DBNull.Value; }
			}

			public string title
			{
				get { return FieldValue<string>("title"); }
				set { this["title"] = value ?? (object)DBNull.Value; }
			}

			public string album
			{
				get { return FieldValue<string>("album"); }
				set { this["album"] = value ?? (object)DBNull.Value; }
			}

			public int length
			{
				get { return FieldValue<int>("length"); }
				set { this["length"] = value; }
			}

			public int track
			{
				get { return FieldValue<int>("track"); }
				set { this["track"] = value; }
			}

			private T FieldValue<T>(string columnName)
			{
				object value = this[columnName];
				if (value == DBNull.Value)
					return default(T);
				return (T)value;
			}
		}
	}

	namespace DataSet1TableAdapters
	{
		public class PlaylistTableAdapter
		{
			private static string PlaylistFile
			{
				get
				{
					return Path.Combine(
						Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
						"CUEPlayer",
						"playlist.xml");
				}
			}

			public int Fill(DataSet1.PlaylistDataTable table)
			{
				table.Clear();
				if (!File.Exists(PlaylistFile))
					return 0;

				DataSet dataSet = new DataSet();
				dataSet.ReadXml(PlaylistFile);
				if (!dataSet.Tables.Contains("Playlist"))
					return 0;

				int count = 0;
				foreach (DataRow row in dataSet.Tables["Playlist"].Rows)
				{
					DataSet1.PlaylistRow playlistRow = table.AddPlaylistRow(
						ReadString(row, "path"),
						ReadString(row, "artist"),
						ReadString(row, "title"),
						ReadString(row, "album"),
						ReadInt32(row, "length"),
						ReadInt32(row, "track"));
					if (dataSet.Tables["Playlist"].Columns.Contains("id"))
						playlistRow.id = ReadInt32(row, "id");
					count++;
				}
				table.AcceptChanges();
				return count;
			}

			public int Update(DataSet1.PlaylistDataTable table)
			{
				string directory = Path.GetDirectoryName(PlaylistFile);
				if (!Directory.Exists(directory))
					Directory.CreateDirectory(directory);

				DataTable copy = table.Clone();
				foreach (DataRow row in table.Rows)
				{
					if (row.RowState == DataRowState.Deleted)
						continue;
					DataRow newRow = copy.NewRow();
					foreach (DataColumn column in table.Columns)
						newRow[column.ColumnName] = row[column, DataRowVersion.Current];
					copy.Rows.Add(newRow);
				}

				DataSet dataSet = new DataSet("DataSet1");
				dataSet.Tables.Add(copy);
				dataSet.WriteXml(PlaylistFile, XmlWriteMode.WriteSchema);
				int count = copy.Rows.Count;
				table.AcceptChanges();
				return count;
			}

			private static string ReadString(DataRow row, string columnName)
			{
				if (!row.Table.Columns.Contains(columnName) || row[columnName] == DBNull.Value)
					return null;
				return Convert.ToString(row[columnName]);
			}

			private static int ReadInt32(DataRow row, string columnName)
			{
				if (!row.Table.Columns.Contains(columnName) || row[columnName] == DBNull.Value)
					return 0;
				return Convert.ToInt32(row[columnName]);
			}
		}
	}
}
