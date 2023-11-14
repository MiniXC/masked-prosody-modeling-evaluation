class ProsodyModelSWBCollator:
    def __init__(
        self,
        args: BURNCollatorArgs, # TODO change?
        device: torch.device = None,
    ):
        """
        Collator for the baseline model, which extracts
        any subset of the following measures from the dataset:
        - pitch
        - energy
        - duration
        - voice activity
        """
        super().__init__()

        self.args = args
        self.vocex = Vocex.from_pretrained(self.args.vocex, fp16=self.args.vocex_fp16)
        self.mpm = MaskedProsodyModel.from_pretrained(self.args.mpm)
        if device is not None:
            self.mpm.to(device)
        self.device = device
        self.bins = torch.linspace(0, 1, self.mpm.args.bins)
        if args.use_algorithmic_features:
            self.pitch_measure = PitchMeasure()
            self.energy_measure = EnergyMeasure()
            self.voice_activity_measure = VoiceActivityMeasure()

    def __call__(self, batch):
        results = []
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        for k, audio in enumerate(batch["audio"]):
            mpms = []
            audio_path = audio
            file = Path(audio).with_suffix(f".mpm.npy")
            if (not file.exists()) or self.args.overwrite:
                file = Path(audio)
                audio, sr = librosa.load(audio, sr=16000)
                # create 6 second windows
                windows = []
                for i in range(0, len(audio), 96000):
                    windows.append(audio[i : i + 96000])
                # Get audio features of entire audio (computed per window)
                for i, w in enumerate(windows):
                    if not self.args.use_algorithmic_features:
                        vocex_output = self.vocex(w, sr)
                        pitch = vocex_output["measures"]["pitch"]
                        energy = vocex_output["measures"]["energy"]
                        vad = vocex_output["measures"]["voice_activity_binary"]
                    else:
                        # resample w to 22050 # TODO ??? 
                        w = librosa.resample(w, orig_sr=sr, target_sr=22050)
                        pitch = self.pitch_measure(w, np.array([1000]))["measure"]
                        energy = self.energy_measure(w, np.array([1000]))["measure"]
                        vad = self.voice_activity_measure(w, np.array([1000]))[
                            "measure"
                        ]
                    # normalize using pitch_min, pitch_max, energy_min, energy_max, vad_min, vad_max
                    pitch = torch.clip(
                        torch.tensor(pitch).unsqueeze(0),
                        self.args.pitch_min,
                        self.args.pitch_max,
                    ) / (self.args.pitch_max - self.args.pitch_min)
                    energy = torch.clip(
                        torch.tensor(energy).unsqueeze(0),
                        self.args.energy_min,
                        self.args.energy_max,
                    ) / (self.args.energy_max - self.args.energy_min)
                    vad = torch.clip(
                        torch.tensor(vad).unsqueeze(0),
                        self.args.vad_min,
                        self.args.vad_max,
                    )
                    # bucketize
                    torch.bucketize(pitch, self.bins) + 2
                    torch.bucketize(energy, self.bins) + 2
                    torch.bucketize(vad, torch.linspace(0, 1, 2)) + 2
                    mpm_input = torch.stack(
                        [
                            pitch,
                            energy,
                            vad,
                        ]
                    ).transpose(0, 1)
                    if self.device is not None:
                        mpm_input = mpm_input.to(self.device)

                    # Slice to max_length, else pad
                    mpm_input = mpm_input[:, :, : self.mpm.args.max_length]
                    prev_len = mpm_input.shape[-1]
                    if mpm_input.shape[-1] < self.mpm.args.max_length:
                        zs = torch.zeros(
                            (
                                mpm_input.shape[0],
                                mpm_input.shape[1],
                                self.mpm.args.max_length - prev_len,
                            )
                        )
                        if self.device is not None:
                            zs = zs.to(self.device)
                        mpm_input = torch.cat(
                            [
                                mpm_input,
                                zs,
                            ],
                            dim=-1,
                        )
                    # NB: self.long() is equivalent to self.to(torch.int64).
                    mpms.append(mpm_input.long())
                mpms_batch = torch.stack(mpms, dim=1).squeeze(0)

                # Get MPM features for each window over entire audio sequence
                mpm_output = self.mpm(mpms_batch, return_layer=self.args.mpm_layer)
                mpms = mpm_output["representations"].detach().cpu()
                # concat to go from (batch_size, len, hidden_dim) to (len, hidden_dim)
                mpms = mpms.reshape(-1, mpms.shape[-1])
                
                # Get word-level representations directly from frame indices
                word_mpms = []
                durations = batch["word_frames"][k]
                for i in range(len(durations) - 1):
                    # mean-max pool over the word
                    word_mpms.append(
                        torch.concat(
                            [
                                mpms[durations[i][0] : durations[i][1]].mean(0),
                                mpms[durations[i][0] : durations[i][1]].max(0).values,
                            ]
                        )
                    )
                word_mpms = torch.stack(word_mpms)
                np.save(
                    Path(audio_path).with_suffix(f".mpm.npy"),
                    word_mpms,
                )
            else:
                word_mpms = np.load(file)
            results.append(word_mpms)

        # pad to max length
        if self.args.max_words is None:
            max_len = np.max([r.shape[0] for r in results])
        else:
            max_len = self.args.max_words

        mask = np.zeros((len(results), max_len))
        for i, r in enumerate(results):
            mask[i, : r.shape[0]] = 1

        batch["mask"] = torch.tensor(mask).bool()

        batch["mpm"] = torch.tensor(
            np.array([np.pad(r, ((0, max_len - r.shape[0]), (0, 0))) for r in results])
        ).to(torch.float32)

        # pad prominence and break
        batch["prominence"] = [np.array(p) for p in batch["prominence"]]
        batch["break"] = [np.array(b) for b in batch["break"]]
        batch["prominence"] = torch.tensor(
            np.array(
                [np.pad(p, (0, max_len - p.shape[0])) for p in batch["prominence"]]
            )
        )
        batch["break"] = torch.tensor(
            np.array([np.pad(b, (0, max_len - b.shape[0])) for b in batch["break"]])
        )
        # pad word durations
        batch["word_durations"] = [np.array(wd) for wd in batch["word_durations"]]
        batch["word_durations"] = torch.tensor(
            np.array(
                [
                    np.pad(wd, (0, max_len - wd.shape[0]))
                    for wd in batch["word_durations"]
                ]
            )
        )

        return batch