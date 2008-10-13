					uint bestTracksMatch = 0;
					int bestWorstConfidence = 0;
					int bestOffset = 0;
					int bestDisk = -1;

					for (int di = 0; di < (int)accDisks.Count; di++)
					{
						for (int offset = -_arOffsetRange; offset <= _arOffsetRange; offset++)
						{
							uint tracksMatch = 0;
							int worstConfidence = -1;

							for (int iTrack = 0; iTrack < TrackCount; iTrack++)
								if (_tracks[iTrack].OffsetedCRC[_arOffsetRange - offset] == accDisks[di].tracks[iTrack].CRC)
								{
									tracksMatch++;
									//confidence += accDisks[di].tracks[iTrack].count;
									if (accDisks[di].tracks[iTrack].CRC != 0)
										if (worstConfidence == -1 || worstConfidence > accDisks[di].tracks[iTrack].count)
											worstConfidence = (int) accDisks[di].tracks[iTrack].count;
										//if (_config.fixWhenConfidence)
										//tracksMatchWithConfidence++;
								}

							if (tracksMatch > bestTracksMatch
								|| (tracksMatch == bestTracksMatch && worstConfidence > bestWorstConfidence)
								|| (tracksMatch == bestTracksMatch && worstConfidence  == bestWorstConfidence && Math.Abs(offset) < Math.Abs(bestOffset))
								)
							{
								bestTracksMatch = tracksMatch;
								bestWorstConfidence = worstConfidence;
								bestOffset = offset;
								bestDisk = di;
							}
						}
					}
					if (bestWorstConfidence > _config.fixWhenConfidence && 
						(bestTracksMatch == TrackCount || 
						(TrackCount > 2 && bestTracksMatch * 100 > TrackCount * _config.fixWhenPercent)))
							_writeOffset = bestOffset;
					else
					if (bestTracksMatch != TrackCount && _config.noUnverifiedOutput)
						SkipOutput = true;















			int s1 = (int) Math.Min(count, Math.Max(0, 450 * 588 - _arOffsetRange - (int)currentOffset));
			int s2 = (int) Math.Min(count, Math.Max(0, 451 * 588 + _arOffsetRange - (int)currentOffset));
			if ( s1 < s2 )
				fixed (uint* FrameCRCs = _tracks[iTrack].OffsetedFrame450CRC)
					for (int sj = s1; sj < s2; sj++)
					{
						int magicFrameOffsetBase = (int)currentOffset + sj - 450 * 588 + 1;
						for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
						{
							int magicFrameOffset = magicFrameOffsetBase + oi;
							if (magicFrameOffset > 0 && magicFrameOffset <= 588)
								FrameCRCs[_arOffsetRange - oi] += (uint)(samples[sj] * magicFrameOffset);
						}
					}


						int magicFrameOffset = (int)currentOffset + sj - 450 * 588 + 1;
						int firstOffset = Math.Max(-_arOffsetRange, 1 - magicFrameOffset);
						int lastOffset = Math.Min(_arOffsetRange, 588 - magicFrameOffset);
						for (int oi = firstOffset; oi <= lastOffset; oi++)
							FrameCRCs[_arOffsetRange - oi] += (uint)(samples[sj] * (magicFrameOffset + oi));
