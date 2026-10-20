# ConfirmModal

`openConfirm` uses the shared keyboard-accessible Headless UI dialog. Existing
callers retain its default layer and empty optional input.

Optional input controls: `inputTitle` is the input placeholder; `inputLabel`
adds a visible label; `initialInputValue` prefills an editor; `inputRequired`
prevents confirmation of blank input. `layer: 'overlay'` places the dialog
above floating application drawers. Saved assistant chats use these options
for title editing and named delete/reload confirmations.
