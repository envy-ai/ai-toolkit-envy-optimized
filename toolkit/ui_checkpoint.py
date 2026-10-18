"""Shared Save Now / pause-at-checkpoint controls for UI trainers."""
from toolkit.basic import flush
from toolkit.print import print_acc


class UICheckpointControls:
    def _checkpoint_flags(self):
        if not getattr(self, 'is_ui_trainer', True):
            return False, False

        def read():
            with self._db_connect() as conn:
                row = conn.execute('SELECT stop, save_now, step FROM Job WHERE id = ?',
                                   (self.job_id,)).fetchone()
                if not row:
                    return False, False
                stop, save = bool(row[0]), bool(row[1])
                # A zero-step pause must not wait for training or save initial
                # weights. Keep resumed jobs protected before their checkpoint
                # step has been restored in memory. Ordinary Save Now is unchanged.
                if stop and self.step_num == 0 and row[2] == 0:
                    save = False
                return stop, save

        retry = getattr(self, '_retry_db_operation', None)
        return retry(read) if retry else read()

    def should_stop(self):
        stop, save = self._checkpoint_flags()
        # A pause is the existing Save Now request plus stop. Do not interrupt
        # loading, sampling, backward, or the save itself while it is pending.
        return stop and not save

    def should_save(self):
        return self._checkpoint_flags()[1]

    def maybe_save(self):
        if getattr(self, 'is_grad_accumulation_step', False):
            return
        stop, save = self._checkpoint_flags()
        if not save:
            return
        if self.progress_bar is not None:
            self.progress_bar.pause()
        self._pause_save_in_progress = stop
        try:
            print_acc(f'\nSaving at step {self.step_num}')
            self.optimizer.zero_grad()
            self.save(self.step_num)
            self.ensure_params_requires_grad()
            flush()
            # Clear only after both weights and optimizer have been written.
            # The caller then handles stop before requesting any more samples.
            self.update_db_key('save_now', 0)
        finally:
            self._pause_save_in_progress = False
            if self.progress_bar is not None:
                self.progress_bar.unpause()
