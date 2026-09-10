using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using System.Windows.Forms;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;

namespace CUERipper
{
    public class ColumnInfo
    {
        public String Name { get; set; }
        public int Width { get; set; }
        public int DisplayIndex { get; set; }
        public ColumnInfo(string name, int width, int displayIndex)
        {
            Name = name;
            Width = width;
            DisplayIndex = displayIndex;
        }
    }



    [Serializable]
    public class DataGridViewColumnSettings : IXmlSerializable
    {
        public List<ColumnInfo> listColumnInfo { get; set; } = new List<ColumnInfo>();

        public void ReadFrom(DataGridView dgv)
        {
            listColumnInfo.Clear();
            foreach (DataGridViewColumn colgrid in dgv.Columns)
            {
                listColumnInfo.Add(new ColumnInfo(colgrid.Name, colgrid.Width, colgrid.DisplayIndex));
            }
            listColumnInfo.Sort((x, y) => x.DisplayIndex.CompareTo(y.DisplayIndex));
        }

        public void ApplyTo(DataGridView dgv)
        {
            int i = 0;
            foreach (ColumnInfo columninf in listColumnInfo)
            {
                var colgrid = dgv.Columns[columninf.Name];
                if (colgrid != null)
                {
                    if (i < dgv.ColumnCount)
                        colgrid.DisplayIndex = i;
                    i++;
                    colgrid.Width = columninf.Width;
                }
            }
        }

        // IXmlSerializable implementation
        public XmlSchema GetSchema() => null;

        public void ReadXml(XmlReader reader)
        {
            listColumnInfo.Clear();

            try
            {
                reader.ReadStartElement();
                int i = 1;
                while (reader.IsStartElement("Column"))
                {
                    string name = reader.GetAttribute("Name");
                    int width = int.Parse(reader.GetAttribute("Width"));
                    ColumnInfo columninf = new ColumnInfo(name, width, i);
                    listColumnInfo.Add(columninf);
                    reader.ReadStartElement("Column");
                    i++;
                }
                reader.ReadEndElement();
            }
            catch (XmlException)
            {
                return;
            }
        }

        public void WriteXml(XmlWriter writer)
        {
            foreach (var columninf in listColumnInfo)
            {
                writer.WriteStartElement("Column");
                writer.WriteAttributeString("Name", columninf.Name);
                writer.WriteAttributeString("Width", columninf.Width.ToString());
                writer.WriteEndElement();
            }
        }
    }

    [XmlRoot("dictionary")]
    public class SerializableDictionary<TKey, TValue>
        : Dictionary<TKey, TValue>, IXmlSerializable
    {
        #region IXmlSerializable Members
        public System.Xml.Schema.XmlSchema GetSchema()
        {
            return null;
        }

        public void ReadXml(System.Xml.XmlReader reader)
        {
            XmlSerializer keySerializer = new XmlSerializer(typeof(TKey));
            XmlSerializer valueSerializer = new XmlSerializer(typeof(TValue));

            bool wasEmpty = reader.IsEmptyElement;
            reader.Read();

            if (wasEmpty)
                return;

            while (reader.NodeType != System.Xml.XmlNodeType.EndElement)
            {
                reader.ReadStartElement("item");

                reader.ReadStartElement("key");
                TKey key = (TKey)keySerializer.Deserialize(reader);
                reader.ReadEndElement();

                reader.ReadStartElement("value");
                TValue value = (TValue)valueSerializer.Deserialize(reader);
                reader.ReadEndElement();

                this.Add(key, value);

                reader.ReadEndElement();
                reader.MoveToContent();
            }
            reader.ReadEndElement();
        }

        public void WriteXml(System.Xml.XmlWriter writer)
        {
            XmlSerializer keySerializer = new XmlSerializer(typeof(TKey));
            XmlSerializer valueSerializer = new XmlSerializer(typeof(TValue));

            foreach (TKey key in this.Keys)
            {
                writer.WriteStartElement("item");

                writer.WriteStartElement("key");
                keySerializer.Serialize(writer, key);
                writer.WriteEndElement();

                writer.WriteStartElement("value");
                TValue value = this[key];
                valueSerializer.Serialize(writer, value);
                writer.WriteEndElement();

                writer.WriteEndElement();
            }
        }
        #endregion
    }

    [Serializable]
    public class CUERipperConfig
    {
        public CUERipperConfig()
        {
            // Iterate through each property and call ResetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
            {
                property.ResetValue(this);
            }

            this.DriveOffsets = new SerializableDictionary<string, int>();
            this.DriveC2ErrorModes = new SerializableDictionary<string, int>();
            this.ReadCDCommands = new SerializableDictionary<string, int>();
        }

        internal static XmlSerializer serializer = new XmlSerializer(typeof(CUERipperConfig));

        [DefaultValue("flac")]
        public string DefaultLosslessFormat { get; set; }

        [DefaultValue("mp3")]
        public string DefaultLossyFormat { get; set; }

        [DefaultValue("lossy.flac")]
        public string DefaultHybridFormat { get; set; }

        public string DefaultDrive { get; set; }

        public SerializableDictionary<string, int> DriveOffsets { get; set; }

        // 0 (None), 1 (Mode294), 2 (Mode296), 3 (Auto)
        public SerializableDictionary<string, int> DriveC2ErrorModes { get; set; }

        // 0 (ReadCdBEh), 1 (ReadCdD8h), 2 (Unknown/AutoDetect)
        public SerializableDictionary<string, int> ReadCDCommands { get; set; }

        public DataGridViewColumnSettings TrackGridSettings { get; set; } = new DataGridViewColumnSettings();

    }
}
