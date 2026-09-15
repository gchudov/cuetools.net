#region Copyright (C) 2026 Max Visser
/*
    Copyright (C) 2026 Max Visser

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
using CUERipper.Avalonia.Compatibility;
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Models.Abstractions;
using CUERipper.Avalonia.Services.Abstractions;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.Processor;
using CUETools.Ripper;
using CUETools.Ripper.Exceptions;
using Microsoft.Extensions.Localization;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Net;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Services
{
    public class DiscRippingService : IDiscRippingService
    {
        private Dictionary<char, DriveInformation> _driveList = [];

        private char _selectedDrive = Constants.NullDrive;
        public char SelectedDrive
        {
            get => _selectedDrive;
            set
            {
                var eventArgs = new DriveChangedEventArgs(_selectedDrive, value);
                _selectedDrive = value;
                OnSelectedDriveChanged?.Invoke(this, eventArgs);
            }
        }

        public event EventHandler<DriveChangedEventArgs>? OnSelectedDriveChanged;
        public event EventHandler<ReadProgressArgs>? OnRippingProgress;
        public event EventHandler<DirectoryConflictEventArgs>? OnDirectoryConflict;

        private readonly ICUEConfigFacade _config;
        private readonly ICDRipperFactory _ripperFactory;
        private readonly ICDDriveEnumerator _driveEnumerator;
        private readonly ICUEMetadataStore _metadataStore;
        private readonly IBitmapFactory _bitmapFactory;
        private readonly IStringLocalizer _localizer;
        private readonly ILogger _logger;

        public DiscRippingService(ICUEConfigFacade config
            , ICDRipperFactory ripperFactory
            , ICDDriveEnumerator driveEnumerator
            , ICUEMetadataStore metadataStore
            , IBitmapFactory bitmapFactory
            , IStringLocalizer<Language> stringLocalizer
            , ILogger<DiscRippingService> logger)
        {
            _config = config;
            _ripperFactory = ripperFactory;
            _driveEnumerator = driveEnumerator;
            _metadataStore = metadataStore;
            _bitmapFactory = bitmapFactory;
            _localizer = stringLocalizer;
            _logger = logger;
        }

        private ICDRipper? CreateCDRipperInstance()
        {
            var cdRipper = _ripperFactory.Create();
            if (cdRipper == null)
            {
                _logger.LogError("Failed to create an instance of CD ripper, is the library missing?");
            }

            return cdRipper;
        }

        private DriveInformation QueryDriveName(char drive)
        {
            var nullResult = new DriveInformation(drive, $"{drive}:", string.Empty, false);

            using var audioSource = CreateCDRipperInstance();
            if (audioSource == null) return nullResult;

            try
            {
                return audioSource.Open(drive)
                    ? new DriveInformation(drive, audioSource.Path, audioSource.ARName, true)
                    : nullResult;
            }
            catch (TOCException)
            {
                // Not clean but it's safe at this point
                return new DriveInformation(drive, audioSource.Path, audioSource.ARName, true);
            }
            catch (ReadCDException ex)
            {
                _logger.LogError(ex, "Failed to read disc '{DriveLetter}'.", drive);
                if (OS.IsWindows() && (uint?)ex.InnerException?.HResult == 0x80070020)
                {
                    var drivePath = $"{audioSource.Path}(Warning: drive is in use)";
                    return new DriveInformation(drive, drivePath, string.Empty, false);
                }

                return new DriveInformation(drive, $"{audioSource.Path}(Error: {ex.Message})", string.Empty, false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to access drive '{DriveLetter}'.", drive);
                return nullResult;
            }
        }

        public IImmutableDictionary<char, DriveInformation> QueryAvailableDriveInformation()
        {
            var result = new Dictionary<char, DriveInformation>();

            var drives = _driveEnumerator.DrivesAvailable();
            foreach (var drive in drives)
            {
                result.Add(drive, QueryDriveName(drive));
            }

            _driveList = result;
            return result.ToImmutableDictionary(x => x.Key, x => x.Value);
        }

        public bool IsDriveAccessible()
            => _driveList.TryGetValue(SelectedDrive, out var result)
                ? result.IsAccessible
                : throw new KeyNotFoundException($"Couldn't find drive key '{SelectedDrive}'.");

        public string GetDriveName()
            => _driveList.TryGetValue(SelectedDrive, out var result)
                ? result.Name
                : throw new KeyNotFoundException($"Couldn't find drive key '{SelectedDrive}'.");

        public string GetDriveARName()
            => _driveList.TryGetValue(SelectedDrive, out var result)
                ? result.ARName
                : throw new KeyNotFoundException($"Couldn't find drive key '{SelectedDrive}'.");

        public CDImageLayout? GetDiscTOC()
        {
            if (!IsDriveAccessible()) return null;

            using var audioSource = CreateCDRipperInstance();
            if (audioSource == null) return null;

            try
            {
                if (!audioSource.Open(SelectedDrive)) return null;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to open drive while trying retrieve TOC.");
                return null;
            }

            return audioSource.TOC;
        }

        public void EjectTray()
        {
            if (!IsDriveAccessible()) return;

            using var audioSource = CreateCDRipperInstance();
            if (audioSource == null) return;

            try
            {
                audioSource.Open(SelectedDrive);
            }
            catch (TOCException)
            {
                // Ignore... We don't care about the TOC here
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to open drive while trying to eject tray.");
                return;
            }

            try
            {
                audioSource.DisableEjectDisc(false);
                audioSource.EjectDisk();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to eject tray.");
                return;
            }
        }

        public int GetDriveOffset()
            => AccurateRipVerify.FindDriveReadOffset(GetDriveARName(), out var driveOffset)
                ? driveOffset
                : 0;

        public Task<CUEResult> RipAsync(RipSettings ripSettings, CancellationToken ct)
        {
            var selectedDrive = SelectedDrive;

            return Task.Factory.StartNew(() =>
            {
                _logger.LogInformation("Rip task has been started.");

                try
                {
                    return Rip(selectedDrive, ripSettings, ct);
                }
                catch (StopException)
                {
                    _logger.LogInformation("Ripping has been stopped by user.");
                    return CUEResult.Failure(_localizer["Status:RipFailUser"], string.Empty);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Ripping has failed! Unexpected error occurred.");
                    return CUEResult.Failure(_localizer["Status:RipFail"]
                        , $"{_localizer["Error:Unexpected"]} {ex.Message}");
                }
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
            // Do not let this task be cancelled by the cancellation token, let it exit cleanly.
        }

        private CUEResult Rip(char selectedDrive
            , RipSettings settings
            , CancellationToken ct)
        {
            if (settings.EncodingConfiguration.None())
            {
                _logger.LogError("Ripping has failed! No encoding configuration found");

                return CUEResult.Failure(_localizer["Status:RipFail"], _localizer["Error:NoEncodingFound"]);
            }

            var initialEncoding = settings.EncodingConfiguration[0];

            if (settings.EncodingConfiguration.Length > 1
                && !initialEncoding.IsLossless)
            {
                _logger.LogError("Ripping has failed! First encoding must be lossless");

                return CUEResult.Failure(_localizer["Status:RipFail"], _localizer["Error:MultiEncodingNotLossless"]);
            }

            _config.ApplyEncodingConfiguration(initialEncoding);

            using var audioSource = CreateCDRipper(selectedDrive, settings, ct);
            if (audioSource == null)
            {
                _logger.LogError("Ripping has failed! Couldn't open audio source on selected drive {selectedDrive}:\\.", selectedDrive);

                return CUEResult.Failure(_localizer["Status:RipFail"], _localizer["Error:RipFailedNoAccessDrive"]);
            }

            var cueSheet = new CUESheet(_config.ToCUEConfig());
            var stopRegistration = ct.Register(() => cueSheet.Stop());
            var ejectDisc = false;

            try
            {
                cueSheet.OpenCD(audioSource);
                cueSheet.Action = CUEAction.Encode;
                cueSheet.UseCUEToolsDB(Constants.ApplicationName, audioSource.ARName, false, _config.MetadataSearch);
                cueSheet.UseAccurateRip();

                General.SetCUELine(cueSheet.Attributes, "REM", "DISCID", AccurateRipVerify.CalculateCDDBId(audioSource.TOC), false);

                CUEMetadataEntry? metadataEntry = GetMetadataEntry(cueSheet, audioSource.TOC, settings.AlbumCoverUri);
                if (metadataEntry == null)
                {
                    return CUEResult.Failure(_localizer["Status:RipFail"], _localizer["Error:RipFailedMetadata"]);
                }

                cueSheet.CopyMetadata(metadataEntry.metadata);

                var encodingFormat = initialEncoding.Encoding;
                var encoderType = initialEncoding.IsLossless
                    ? AudioEncoderType.Lossless
                    : AudioEncoderType.Lossy;

                cueSheet.OutputStyle = initialEncoding.CUEStyleIndex == 0
                        ? _config.CanEmbedCUE(initialEncoding) ? CUEStyle.SingleFileWithCUE : CUEStyle.SingleFile
                        : CUEStyle.GapsAppended;

                string pathOut = cueSheet.GenerateUniqueOutputPath(_config.PathFormat,
                        cueSheet.OutputStyle == CUEStyle.SingleFileWithCUE ? "." + encodingFormat : Constants.CueExtension,
                        CUEAction.Encode, null);

                if (string.IsNullOrWhiteSpace(pathOut))
                {
                    _logger.LogError("Ripping has failed! Couldn't generate the output path.");

                    return CUEResult.Failure(_localizer["Status:RipFail"], _localizer["Error:RipFailedOutputPath"]);
                }

                if (Directory.Exists(Path.GetDirectoryName(pathOut)
                        ?? throw new DirectoryNotFoundException(pathOut)))
                {
                    var eventArgs = new DirectoryConflictEventArgs(pathOut, false);
                    OnDirectoryConflict?.Invoke(this, eventArgs);

                    if (!eventArgs.CanModifyContent)
                    {
                        _logger.LogError("Ripping has failed! Couldn't generate the output path. Directory already exists.");

                        return CUEResult.Failure(_localizer["Status:RipFail"], _localizer["Error:RipFailedOutputPath"]);
                    }
                }

                if (string.IsNullOrWhiteSpace(cueSheet.Metadata.Comment))
                {
                    cueSheet.Metadata.Comment = audioSource.RipperVersion;
                }

                cueSheet.GenerateFilenames(encoderType, encodingFormat, pathOut);

                CopyRawAlbumCoverFromCache(settings.AlbumCoverUri, pathOut);

                if (_config.DisableEjectDisc)
                {
                    _logger.LogInformation("Disabling disc ejecting.");
                    audioSource.DisableEjectDisc(true);
                }

                if (settings.TestAndCopy)
                {
                    _logger.LogInformation("Testing before copy.");
                    cueSheet.TestBeforeCopy();
                }
                else
                {
                    cueSheet.ArTestVerify = null;
                }

                _logger.LogInformation("Ripping has started.");

                cueSheet.Go();

                _logger.LogInformation("Ripping has finished.");

                ejectDisc = _config.EjectAfterRip;

#if !DEBUG
                _logger.LogInformation("Submitting to CUETools Database.");

                cueSheet.CTDB.Submit(
                    (int)cueSheet.ArVerify.WorstConfidence() + 1,
                    audioSource.CorrectionQuality == 0 ? 0 :
                    (int)(100 * (1.0 - Math.Log(audioSource.FailedSectors.PopulationCount() + 1) / Math.Log(audioSource.TOC.AudioLength + 1))),
                    cueSheet.Metadata.Artist,
                    cueSheet.Metadata.Title,
                    cueSheet.TOC.Barcode);
#endif

                RipStatus status;
                string statusText = string.Empty;
                string popupContent = cueSheet.GenerateVerifyStatus() + ".";

                if (audioSource.FailedSectors.PopulationCount() != 0)
                {
                    if (settings.EncodingConfiguration[0].IsLossless
                            && cueSheet.CTDB.QueryExceptionStatus == WebExceptionStatus.Success)
                    {
                        status = cueSheet.CTDB.Entries.Any(x => x.canRecover && x.hasErrors)
                            ? RipStatus.Repairable : RipStatus.CompletedWithErrors;
                    }
                    else
                    {
                        status = RipStatus.CompletedWithErrors;
                    }

                    statusText = _localizer["Warning:RipTroubledDisc"];
                }
                else
                {
                    status = RipStatus.Completed;
                    statusText = _localizer["Status:RipFinished"];
                }

                return new CUEResult(status
                    , statusText
                    , popupContent
                    , pathOut);
            }
            finally
            {
                _logger.LogInformation("Read command: {ReadCommand}", audioSource.CurrentReadCommand);

                stopRegistration.Dispose();

                TryCleanup(cueSheet.Close, "Failed to close the cue sheet.");

                if (_config.DisableEjectDisc || ejectDisc)
                {
                    _logger.LogInformation("Enabling disc ejecting.");
                    TryCleanup(() => audioSource.DisableEjectDisc(false), "Failed to re-enable disc ejecting.");
                }

                if (ejectDisc)
                {
                    _logger.LogInformation("Ejecting disc from drive.");

                    TryCleanup(audioSource.EjectDisk, "Failed to eject the disc after ripping.");
                }
            }
        }

        private void TryCleanup(Action action, string failureMessage)
        {
            try
            {
                action();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, failureMessage);
            }
        }

        private ICDRipper? CreateCDRipper(char selectedDrive, RipSettings ripSettings, CancellationToken ct)
        {
            var audioSource = CreateCDRipperInstance();
            if (audioSource == null) return null;

            try
            {
                if (!audioSource.Open(selectedDrive))
                {
                    audioSource.Dispose();
                    return null;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to open audio source for drive {Drive}.", selectedDrive);

                audioSource.Dispose();
                return null;
            }

            audioSource.DriveOffset = ripSettings.DriveOffset;
            audioSource.DriveC2ErrorMode = (int)ripSettings.C2ErrorModeSetting;
            audioSource.CorrectionQuality = ripSettings.CorrectionQuality;

            audioSource.ReadProgress += (object? sender, ReadProgressArgs args) =>
            {
                // Without throwing the StopException, the application will crash because it'll try
                // to continue reading while the audioSource has been disposed.
                if (ct.IsCancellationRequested) throw new StopException();
                OnRippingProgress?.Invoke(sender, args);
            };

            return audioSource;
        }

        private CUEMetadataEntry? GetMetadataEntry(CUESheet cueSheet
            , CDImageLayout TOC
            , string albumCoverUri)
        {
            try
            {
                CUEMetadata? cache = _metadataStore.Load(TOC.TOCID);
                if (cache == null) return null;

                var metadataEntry = new CUEMetadataEntry(cache, TOC, "local");

                using var albumCover = GetAlbumCoverFromCache(albumCoverUri);
                if (albumCover != null)
                {
                    IBitmap? embeddedArtwork = null;
                    if (albumCover.Width > _config.MaxAlbumArtSize
                        || albumCover.Height > _config.MaxAlbumArtSize)
                    {
                        embeddedArtwork = albumCover.ContainedResize(_config.MaxAlbumArtSize);
                    }

                    byte[] byteArray = [];
                    using (var stream = new MemoryStream())
                    {
                        (embeddedArtwork ?? albumCover).SaveJpeg(stream, Constants.JpegQuality);
                        byteArray = stream.ToArray();
                    }

                    embeddedArtwork?.Dispose();

                    metadataEntry.cover = byteArray;

                    if (_config.EmbedAlbumArt)
                    {
                        var blob = new TagLib.ByteVector(metadataEntry.cover);
                        cueSheet.AlbumArt.Add(new TagLib.Picture(blob) { Type = TagLib.PictureType.FrontCover });
                    }
                }

                return metadataEntry;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Ripping has failed! Couldn't load album metadata.");
                return null;
            }
        }

        private IBitmap? GetAlbumCoverFromCache(string coverUri)
        {
            if (string.IsNullOrWhiteSpace(coverUri)) return null;

            using var md5 = MD5.Create();
            var fileIdentifier = md5.ComputeHashAsString(coverUri);
            var filePath = Path.Combine(Constants.PathImageCache, $"{fileIdentifier}{Constants.JpgExtension}");
            return File.Exists(filePath) ? _bitmapFactory.FromFile(filePath) : null;
        }

        private static void CopyRawAlbumCoverFromCache(string coverUri, string destination)
        {
            if (string.IsNullOrWhiteSpace(coverUri)
                || string.IsNullOrWhiteSpace(destination)) return;

            var outputFolder = Path.GetDirectoryName(destination) ?? throw new DirectoryNotFoundException(destination);

            using var md5 = MD5.Create();
            var fileIdentifier = md5.ComputeHashAsString(coverUri);
            var filePath = Path.Combine(Constants.PathImageCache, $"{fileIdentifier}{Constants.JpgExtension}");

            if (File.Exists(filePath))
            {
                Directory.CreateDirectory(outputFolder);
                File.Copy(filePath, Path.Combine(outputFolder, $"{Constants.HiResCoverName}{Constants.JpgExtension}"), true);
            }
        }
    }
}
