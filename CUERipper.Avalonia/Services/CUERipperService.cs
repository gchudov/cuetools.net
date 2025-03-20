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
using Avalonia.Media.Imaging;
using CUERipper.Avalonia.Compatibility;
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Services.Abstractions;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.CTDB;
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
    public class CUERipperService : ICUERipperService
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
        public event EventHandler<CUEToolsProgressEventArgs>? OnSecondaryProgress;
        public event EventHandler<CUEToolsSelectionEventArgs>? OnRepairSelection;
        public event EventHandler<ReadProgressArgs>? OnRippingProgress;
        public event EventHandler<RipperFinishedEventArgs>? OnFinish;
        public event EventHandler<DirectoryConflictEventArgs>? OnDirectoryConflict;

        private readonly ICUEConfigFacade _config;
        private readonly IStringLocalizer _localizer;
        private readonly ILogger _logger;
        public CUERipperService(ICUEConfigFacade config
            , IStringLocalizer<Language> stringLocalizer
            , ILogger<CUERipperService> logger)
        {
            _config = config;
            _localizer = stringLocalizer;
            _logger = logger;
        }

        private static ICDRipper CreateCDRipperInstance()
            => Activator.CreateInstance(CUEProcessorPlugins.ripper) as ICDRipper
                ?? throw new NullReferenceException("Failed to create instance of CD ripper.");

        private DriveInformation QueryDriveName(char drive)
        {
            using var audioSource = CreateCDRipperInstance();

            var nullResult = new DriveInformation(drive, $"{drive}:", string.Empty, false);
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

            var drives = CDDrivesList.DrivesAvailable();
            foreach(var drive in drives)
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

        public Task RipAudioTracks(RipSettings ripSettings, CancellationToken ct)
        {
            var selectedDrive = SelectedDrive;

            return Task.Factory.StartNew(() =>
            {
                if (ripSettings.EncodingConfiguration.None())
                {
                    _logger.LogError("Ripping has failed! No encoding configuration found");

                    OnFinish?.Invoke(this, new(false, _localizer["Status:RipFail"], _localizer["Error:NoEncodingFound"]));
                    return;
                }

                if (ripSettings.EncodingConfiguration.Length > 1 
                    && !ripSettings.EncodingConfiguration[0].IsLossless)
                {
                    _logger.LogError("Ripping has failed! First encoding must be lossless");

                    OnFinish?.Invoke(this, new(false, _localizer["Status:RipFail"], _localizer["Error:MultiEncodingNotLossless"]));
                    return;
                }

                SetEncodingVariables(ripSettings.EncodingConfiguration[0]);

                using var audioSource = CreateCDRipper(selectedDrive, ripSettings, ct);
                if (audioSource == null)
                {
                    _logger.LogError("Ripping has failed! Couldn't open audio source on selected drive {selectedDrive}:\\.", selectedDrive);

                    OnFinish?.Invoke(this, new(false, _localizer["Status:RipFail"], _localizer["Error:RipFailedNoAccessDrive"]));
                    return;
                }

                var cueSheet = new CUESheet(_config.ToCUEConfig());
                ct.Register(() => cueSheet.Stop());

                cueSheet.OpenCD(audioSource);
                cueSheet.Action = CUEAction.Encode;
                cueSheet.UseCUEToolsDB(Constants.ApplicationName, audioSource.ARName, false, _config.MetadataSearch);
                cueSheet.UseAccurateRip();

                General.SetCUELine(cueSheet.Attributes, "REM", "DISCID", AccurateRipVerify.CalculateCDDBId(audioSource.TOC), false);

                CUEMetadataEntry? metadataEntry = GetMetadataEntry(cueSheet, audioSource.TOC, ripSettings.AlbumCoverUri);
                if (metadataEntry == null)
                {
                    OnFinish?.Invoke(this, new(false, _localizer["Status:RipFail"], _localizer["Error:RipFailedMetadata"]));

                    cueSheet.Close();
                    return;
                }

                cueSheet.CopyMetadata(metadataEntry.metadata);

                var encodingFormat = ripSettings.EncodingConfiguration[0].Encoding;
                var encoderType = ripSettings.EncodingConfiguration[0].IsLossless
                    ? AudioEncoderType.Lossless
                    : AudioEncoderType.Lossy;

                cueSheet.OutputStyle = ripSettings.EncodingConfiguration[0].CUEStyleIndex == 0
                        ? CUEStyle.SingleFileWithCUE
                        : CUEStyle.GapsAppended;

                string pathOut = cueSheet.GenerateUniqueOutputPath(_config.PathFormat,
                        cueSheet.OutputStyle == CUEStyle.SingleFileWithCUE ? "." + encodingFormat : Constants.CueExtension,
                        CUEAction.Encode, null);

                if (string.IsNullOrWhiteSpace(pathOut))
                {
                    _logger.LogError("Ripping has failed! Couldn't generate the output path.");

                    OnFinish?.Invoke(this, new(false, _localizer["Status:RipFail"], _localizer["Error:RipFailedOutputPath"]));

                    cueSheet.Close();
                    return;
                }

                if (Directory.Exists(Path.GetDirectoryName(pathOut)
                        ?? throw new DirectoryNotFoundException(pathOut)))
                {
                    var eventArgs = new DirectoryConflictEventArgs(pathOut, false);
                    OnDirectoryConflict?.Invoke(this, eventArgs);

                    if (!eventArgs.CanModifyContent)
                    {
                        _logger.LogError("Ripping has failed! Couldn't generate the output path. Directory already exists.");

                        OnFinish?.Invoke(this, new(false, _localizer["Status:RipFail"], _localizer["Error:RipFailedOutputPath"]));

                        cueSheet.Close();
                        return;

                    }
                }

                if (string.IsNullOrWhiteSpace(cueSheet.Metadata.Comment))
                {
                    cueSheet.Metadata.Comment = audioSource.RipperVersion;
                }

                cueSheet.GenerateFilenames(encoderType, encodingFormat, pathOut);

                CopyRawAlbumCoverFromCache(ripSettings.AlbumCoverUri, pathOut);

                try
                {
                    if (_config.DisableEjectDisc)
                        audioSource.DisableEjectDisc(true);

                    if (ripSettings.TestAndCopy) cueSheet.TestBeforeCopy();
                    else cueSheet.ArTestVerify = null;

                    cueSheet.Go();

                    _logger.LogInformation("Ripping has finished.");

    #if !DEBUG
                    cueSheet.CTDB.Submit(
                        (int)cueSheet.ArVerify.WorstConfidence() + 1,
                        audioSource.CorrectionQuality == 0 ? 0 :
                        (int)(100 * (1.0 - Math.Log(audioSource.FailedSectors.PopulationCount() + 1) / Math.Log(audioSource.TOC.AudioLength + 1))),
                        cueSheet.Metadata.Artist,
                        cueSheet.Metadata.Title,
                        cueSheet.TOC.Barcode);
    #endif

                    bool recoveryPossible = false;
                    if (cueSheet.CTDB.QueryExceptionStatus == WebExceptionStatus.Success && audioSource.FailedSectors.PopulationCount() != 0)
                    {
                        foreach (DBEntry entry in cueSheet.CTDB.Entries)
                        {
                            recoveryPossible = entry.hasErrors && entry.canRecover;
                            break;
                        }
                    }

                    if (audioSource.FailedSectors.PopulationCount() != 0)
                    {
                        if (recoveryPossible && !_config.SkipRepair)
                        {
                            var repairCue = RepairTracks(encoderType, encodingFormat, cueSheet.OutputStyle, metadataEntry, pathOut, ct);
                            if (repairCue != null)
                            {
                                cueSheet.Close();
                                cueSheet = repairCue;
                                recoveryPossible = false;
                            }
                        }

                        EncodeTracksPerConfig(pathOut, ripSettings.EncodingConfiguration, metadataEntry, ct);

                        OnFinish?.Invoke(this, new(true, _localizer["Warning:RipTroubledDisc"], cueSheet.GenerateVerifyStatus() + "."));
                    }
                    else
                    {
                        EncodeTracksPerConfig(pathOut, ripSettings.EncodingConfiguration, metadataEntry, ct);

                        if (_config.AutomaticRip)
                            OnFinish?.Invoke(this, new(true, _localizer["Status:RipFinished"], string.Empty));
                        else
                            OnFinish?.Invoke(this, new(true, _localizer["Status:RipFinished"], cueSheet.GenerateVerifyStatus() + "."));
                    }
                }
                catch (StopException)
                {
                    _logger.LogInformation("Ripping has been stopped by user.");

                    OnFinish?.Invoke(this, new(false, _localizer["Status:RipFailUser"], string.Empty));
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Ripping has failed! Unexpected error occurred.");

                    OnFinish?.Invoke(this, new(false, _localizer["Status:RipFail"], $"{_localizer["Error:Unexpected"]} {ex.Message}"));
                }
                finally
                {
                    cueSheet.Close();

                    if (_config.DisableEjectDisc)
                        audioSource.DisableEjectDisc(false);

                    if (_config.EjectAfterRip)
                        EjectTray();
                }
            }, ct, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }

        private ICDRipper? CreateCDRipper(char selectedDrive, RipSettings ripSettings, CancellationToken ct)
        {
            var audioSource = CreateCDRipperInstance();
            if (!audioSource.Open(selectedDrive))
            {
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
                CUEMetadata cache = CUEMetadata.Load(TOC.TOCID);
                if (cache == null) return null;

                var metadataEntry = new CUEMetadataEntry(cache, TOC, "local");

                using var albumCover = GetAlbumCoverFromCache(albumCoverUri);
                if (albumCover != null)
                {
                    Bitmap? embeddedArtwork = null; 
                    if (albumCover.PixelSize.Width > _config.MaxAlbumArtSize
                        || albumCover.PixelSize.Height > _config.MaxAlbumArtSize)
                    {
                        embeddedArtwork = albumCover.ContainedResize(_config.MaxAlbumArtSize);
                    }

                    byte[] byteArray = [];
                    using (var stream = new MemoryStream())
                    {
                        (embeddedArtwork ?? albumCover).Save(stream, quality: 95);
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

        private static Bitmap? GetAlbumCoverFromCache(string coverUri)
        {
            if (string.IsNullOrWhiteSpace(coverUri)) return null;

            using var md5 = MD5.Create();
            var fileIdentifier = md5.ComputeHashAsString(coverUri);
            var filePath = Path.Combine(Constants.PathImageCache, $"{fileIdentifier}{Constants.JpgExtension}");
            return File.Exists(filePath) ? new Bitmap(filePath) : null;
        }

        private static void CopyRawAlbumCoverFromCache(string coverUri, string destination)
        {
            if (string.IsNullOrWhiteSpace(coverUri)
                || string.IsNullOrWhiteSpace(destination)) return;

            var outputFolder = Path.GetDirectoryName(destination) ?? throw new DirectoryNotFoundException(destination);

            using var md5 = MD5.Create();
            var fileIdentifier = md5.ComputeHashAsString(coverUri);
            var filePath = Path.Combine(Constants.PathImageCache, $"{fileIdentifier}{Constants.JpgExtension}");
            
            if(File.Exists(filePath))
            {
                Directory.CreateDirectory(outputFolder);
                File.Copy(filePath, Path.Combine(outputFolder, $"{Constants.HiResCoverName}{Constants.JpgExtension}"), true);
            }
        }

        private CUESheet? RepairTracks(AudioEncoderType encoderType, string encodingFormat, CUEStyle cueStyle, CUEMetadataEntry metaEntry, string cuePath, CancellationToken ctx)
        {
            var cueSheet = new CUESheet(_config.ToCUEConfig())
            {
                Action = CUEAction.Encode,
                OutputStyle = cueStyle,
            };

            cueSheet.CUEToolsProgress += (object? sender, CUEToolsProgressEventArgs args) =>
            {
                if (ctx.IsCancellationRequested) throw new StopException();
                OnSecondaryProgress?.Invoke(sender, args);
            };

            cueSheet.CUEToolsSelection += (object? sender, CUEToolsSelectionEventArgs args) =>
            {
                OnRepairSelection?.Invoke(sender, args);
            };

            cueSheet.Open(cuePath);
            cueSheet.CopyMetadata(metaEntry.metadata);

            cueSheet.UseAccurateRip();

            string cueDirectory = Path.GetDirectoryName(cuePath) ?? throw new DirectoryNotFoundException(cuePath);
            string cueFileName = Path.GetFileName(cuePath);
            string repairPath = $"{cueDirectory}/{Constants.TempFolderCUERipper}";
            string repairCuePath = $"{repairPath}/{cueFileName}";

            if (Directory.Exists(repairPath))
            {
                Directory.Delete(repairPath, true);
            }

            cueSheet.GenerateFilenames(encoderType, encodingFormat, repairCuePath);

            const string REPAIR_SCRIPT = "repair";
            if (!_config.Scripts.TryGetValue(REPAIR_SCRIPT, out CUEToolsScript? value))
            {
                _logger.LogError("Where did the repair script go?");
                throw new CUEToolsCoreException("For some reason the repair script seems to be missing?");
            }

            try
            {
                cueSheet.ExecuteScript(value);

                if (!Directory.Exists(repairPath))
                {
                    // Repair cancelled
                    return null;
                }

                foreach (string source in Directory.GetFiles(repairPath))
                {
                    string fileName = Path.GetFileName(source);
                    string destination = Path.Combine(cueDirectory, fileName);

                    if (File.Exists(destination))
                    {
                        File.Delete(destination);
                    }

                    File.Move(source, destination);
                }
            }
            finally
            {
                if (Directory.Exists(repairPath))
                {
                    Directory.Delete(repairPath, true);
                }
            }

            return cueSheet;
        }

        private void EncodeTracksPerConfig(string cuePath
            , EncodingConfiguration[] encodingConfiguration
            , CUEMetadataEntry metadataEntry
            , CancellationToken ct)
        {
            string cueDirectory = Path.GetDirectoryName(cuePath) ?? throw new DirectoryNotFoundException(cuePath);
            string cueFileName = Path.GetFileName(cuePath);

            // Skip the first, because it's already encoded :)
            for (int i = 1; i < encodingConfiguration.Length; ++i)
            {
                if (ct.IsCancellationRequested) break;

                var encodingConfig = encodingConfiguration[i];
                SetEncodingVariables(encodingConfig);

                string destination = $"{cueDirectory}/{i}-{encodingConfig.Encoding}";
                string destinationCuePath = $"{destination}/{cueFileName}";

                EncodeTracks(cuePath, destination, destinationCuePath, encodingConfig, metadataEntry, i, encodingConfiguration.Length - 1, ct);
            }

            SetEncodingVariables(encodingConfiguration[0]);
        }

        private void EncodeTracks(string source
            , string destination
            , string destinationCue
            , EncodingConfiguration encodingConfig
            , CUEMetadataEntry metadataEntry
            , int current
            , int total
            , CancellationToken ct)
        {
            var cueSheet = new CUESheet(_config.ToCUEConfig())
            {
                Action = CUEAction.Encode,
                OutputStyle = encodingConfig.CUEStyleIndex == 0
                        ? CUEStyle.SingleFileWithCUE
                        : CUEStyle.GapsAppended
            };
            
            cueSheet.CUEToolsProgress += (object? sender, CUEToolsProgressEventArgs args) =>
            {
                if (ct.IsCancellationRequested) throw new StopException();

                args.status = $"({current}/{total}) {args.status}";

                OnSecondaryProgress?.Invoke(sender, args);
            };

            cueSheet.Open(source);
            cueSheet.CopyMetadata(metadataEntry.metadata);

            var encoderType = encodingConfig.IsLossless
                ? AudioEncoderType.Lossless
                : AudioEncoderType.Lossy;

            if (encoderType == AudioEncoderType.Lossless) cueSheet.UseAccurateRip();

            if (Directory.Exists(destination))
            {
                Directory.Delete(destination, true);
            }

            cueSheet.GenerateFilenames(encoderType, encodingConfig.Encoding, destinationCue);

            bool isSuccess = false;

            try
            {
                cueSheet.Go();

                isSuccess = true;
            }
            finally
            {
                if (!isSuccess && Directory.Exists(destination))
                {
                    Directory.Delete(destination, true);
                }

                cueSheet.Close();
            }
        }

        private void SetEncodingVariables(EncodingConfiguration encodingConfig)
        {
            var currentEncoding = _config.Formats
                .Where(f => f.Key == encodingConfig.Encoding)
                .Select(e => e.Value)
                .Single();

            var requestedEncoder = _config.Encoders
                .Where(e => string.Compare(e.Extension, encodingConfig.Encoding, true) == 0)
                .Where(e => string.Compare(e.Name, encodingConfig.Encoder, true) == 0)
                .Single();

            requestedEncoder.Settings.EncoderMode = encodingConfig.EncoderMode;
            _config.CUEStyleIndex = encodingConfig.CUEStyleIndex;

            if (encodingConfig.IsLossless)
            {
                currentEncoding.encoderLossless = requestedEncoder;
                _config.DefaultLosslessFormat = encodingConfig.Encoding;
                _config.OutputCompression = AudioEncoderType.Lossless;
            }
            else
            {
                currentEncoding.encoderLossy = requestedEncoder;
                _config.DefaultLossyFormat = encodingConfig.Encoding;
                _config.OutputCompression = AudioEncoderType.Lossy;
            }
        }
    }
}
