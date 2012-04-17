using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;

namespace CUETools.Processor
{
    public class Tagging
    {
        public static bool UpdateTags(TagLib.File fileInfo, NameValueCollection tags, CUEConfig config)
        {
            if (fileInfo is TagLib.Riff.File)
                return false;
            TagLib.Ogg.XiphComment xiph = (TagLib.Ogg.XiphComment)fileInfo.GetTag(TagLib.TagTypes.Xiph);
            if (xiph != null)
            {
                foreach (string tag in tags.AllKeys)
                    xiph.SetField(tag, tags.GetValues(tag));
                return true;
            }
            if (fileInfo is TagLib.Mpeg4.File)
            {
                var mpeg4 = (TagLib.Mpeg4.AppleTag)fileInfo.GetTag(TagLib.TagTypes.Apple, true);
                foreach (string tag in tags.AllKeys)
                {
                    mpeg4.SetDashBox("com.apple.iTunes", tag, string.Join(";", tags.GetValues(tag)));
                }
                return true;
            }
            if (fileInfo is TagLib.Mpeg.AudioFile || (fileInfo is TagLib.UserDefined.File && (fileInfo as TagLib.UserDefined.File).Tagger == CUEToolsTagger.ID3v2))
            {
                var id3v2 = (TagLib.Id3v2.Tag)fileInfo.GetTag(TagLib.TagTypes.Id3v2, true);
                id3v2.Version = (byte) (config.advanced.UseId3v24 ? 4 : 3);
                foreach (string tag in tags.AllKeys)
                {
                    var frame = TagLib.Id3v2.UserTextInformationFrame.Get(id3v2, tag, true);
                    frame.Text = tags.GetValues(tag);
                }
                return true;
            }
            TagLib.Ape.Tag ape = (TagLib.Ape.Tag)fileInfo.GetTag(TagLib.TagTypes.Ape, true);
            foreach (string tag in tags.AllKeys)
                ape.SetValue(XiphTagNameToApe(tag), tags.GetValues(tag));
            return true;
        }

        public static void UpdateTags(string path, NameValueCollection tags, CUEConfig config)
        {
            TagLib.UserDefined.AdditionalFileTypes.Config = config;
            TagLib.File fileInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(path));
            if (UpdateTags(fileInfo, tags, config))
                fileInfo.Save();
            //IAudioSource audioSource = AudioReadWrite.GetAudioSource(path, null, config);
            //audioSource.Tags = tags;
            //audioSource.UpdateTags(false);
            //audioSource.Close();
            //audioSource = null;
        }

        public static string[] GetMiscTag(TagLib.File file, string name)
        {
            //TagLib.Mpeg4.AppleTag apple = (TagLib.Mpeg4.AppleTag)file.GetTag(TagLib.TagTypes.Apple);
            //TagLib.Id3v2.Tag id3v2 = (TagLib.Id3v2.Tag)file.GetTag(TagLib.TagTypes.Id3v2);
            TagLib.Ogg.XiphComment xiph = (TagLib.Ogg.XiphComment)file.GetTag(TagLib.TagTypes.Xiph);
            TagLib.Ape.Tag ape = (TagLib.Ape.Tag)file.GetTag(TagLib.TagTypes.Ape);

            //if (apple != null)
            //{
            //    string[] text = apple.GetText(name);
            //    if (text.Length != 0)
            //        return text;
            //}

            //if (id3v2 != null)
            //    foreach (TagLib.Id3v2.Frame f in id3v2.GetFrames())
            //        if (f is TagLib.Id3v2.TextInformationFrame && ((TagLib.Id3v2.TextInformationFrame)f).Text != null)
            //            return ((TagLib.Id3v2.TextInformationFrame)f).Text;

            if (xiph != null)
            {
                string[] l = xiph.GetField(name);
                if (l != null && l.Length != 0)
                    return l;
            }

            if (ape != null)
            {
                TagLib.Ape.Item item = ape.GetItem(name);
                if (item != null)
                    return item.ToStringArray();
            }

            return null;
        }

        public static string TagListToSingleValue(string[] list)
        {
            return list == null ? null :
                list.Length == 0 ? null :
                list.Length == 1 ? list[0] :
                null; // TODO: merge them?
        }

        public static string ApeTagNameToXiph(string tag)
        {
            if (tag.ToUpper() == "YEAR")
                return "DATE";
            if (tag.ToUpper() == "TRACK")
                return "TRACKNUMBER";
            if (tag.ToUpper() == "DISC")
                return "DISCNUMBER";
            return tag;
        }

        public static string XiphTagNameToApe(string tag)
        {
            if (tag.ToUpper() == "DATE")
                return "Year";
            if (tag.ToUpper() == "TRACKNUMBER")
                return "Track";
            if (tag.ToUpper() == "DISCNUMBER")
                return "Disc";
            return tag;
        }

        public static NameValueCollection Analyze(string path)
        {
            return Analyze(new TagLib.File.LocalFileAbstraction(path));
        }

        public static NameValueCollection Analyze(TagLib.File.IFileAbstraction file)
        {
            return Analyze(TagLib.File.Create(file));
        }

        public static NameValueCollection Analyze(TagLib.File fileInfo)
        {
            NameValueCollection tags = new NameValueCollection();

            TagLib.Ogg.XiphComment xiph = (TagLib.Ogg.XiphComment)fileInfo.GetTag(TagLib.TagTypes.Xiph);
            TagLib.Ape.Tag ape = (TagLib.Ape.Tag)fileInfo.GetTag(TagLib.TagTypes.Ape);

            if (xiph != null)
            {
                foreach (string tag in xiph)
                    foreach (string value in xiph.GetField(tag))
                        tags.Add(tag, value);
            }
            else if (ape != null)
            {
                foreach (string tag in ape)
                    foreach (string value in ape.GetItem(tag).ToStringArray())
                        tags.Add(ApeTagNameToXiph(tag), value);
            }
            else
            {
                //if (audioSource is CUETools.Codecs.ALAC.ALACReader)
                //tags = (audioSource as CUETools.Codecs.ALAC.ALACReader).Tags;
            }

            // TODO: enumerate dash atoms somehow?
            //TagLib.Mpeg4.AppleTag apple = (TagLib.Mpeg4.AppleTag)fileInfo.GetTag(TagLib.TagTypes.Apple);
            //if (apple != null)
            //{
            //    tags = new NameValueCollection();
            //    foreach (TagLib.Mpeg4.Box tag in apple)
            //        if (tag.BoxType == "----")
            //            foreach (string value in apple.GetDashBox(tag.)
            //                tags.Add(tag, value);
            //}
            return tags;
        }

        //public void SetTextField(TagLib.File file,
        //    TagLib.ByteVector apple_name, TagLib.ByteVector id3v2_name,
        //    string xiph_name, string ape_name, string[] values)
        //{
        //    TagLib.Mpeg4.AppleTag apple = (TagLib.Mpeg4.AppleTag)file.GetTag(TagLib.TagTypes.Apple, true);
        //    TagLib.Id3v2.Tag id3v2 = (TagLib.Id3v2.Tag)file.GetTag(TagLib.TagTypes.Id3v2, true);
        //    TagLib.Ogg.XiphComment xiph = (TagLib.Ogg.XiphComment)file.GetTag(TagLib.TagTypes.Xiph, true);
        //    TagLib.Ape.Tag ape = (TagLib.Ape.Tag)file.GetTag(TagLib.TagTypes.Ape, (file is TagLib.Mpc.File));

        //    if (apple != null)
        //        apple.SetText(apple_name, values);

        //    if (id3v2 != null)
        //        id3v2.SetTextFrame(id3v2_name, new TagLib.StringList(values));

        //    if (xiph != null)
        //        xiph.AddFields(xiph_name, values);

        //    if (ape != null)
        //        ape.AddValues(ape_name, values, true);
        //}
    }
}
