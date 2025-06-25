﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Xml.Serialization;

namespace CUETools.Processor
{
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

        public static XmlSerializer serializer = new XmlSerializer(typeof(CUERipperConfig));

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

        /// <summary>
        /// Automatically initiates ripping and bypasses non-error popups. (CUERipper new)
        /// </summary>
        public bool AutomaticRip { get; set; }

        /// <summary>
        /// Stops the repair process upon detecting inaccuracies in the rip. (CUERipper new)
        /// </summary>
        public bool SkipRepair { get; set; }

        /// <summary>
        /// Indicates whether the detail pane was open in the last recorded state. (CUERipper new)
        /// </summary>
        public bool DetailPaneOpened { get; set; }

        /// <summary>
        /// JSON-string containing encoding configuration. (CUERipper new)
        /// </summary>
        public string EncodingConfiguration { get; set; }
    }
}
