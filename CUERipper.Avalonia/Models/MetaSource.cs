#region Copyright (C) 2025 Max Visser
/*
    Copyright (C) 2025 Max Visser

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, see <https://www.gnu.org/licenses/>.
*/
#endregion
namespace CUERipper.Avalonia.Models
{
    public enum MetaSource
    {
        None
        , Local
        , MusicBrainz
        , Discogs
        , Freedb
    }

    public static class MetaSourceHelper 
    {
        public static MetaSource FromString(string? str)
            => str?.ToLower() switch {
                "local" => MetaSource.Local
                , "musicbrainz" => MetaSource.MusicBrainz
                , "discogs" => MetaSource.Discogs
                , "freedb" => MetaSource.Freedb
                , _ => MetaSource.None
            };
    }
}