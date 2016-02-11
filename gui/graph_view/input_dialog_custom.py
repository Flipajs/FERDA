QDialog dialog(this);
// Use a layout allowing to have a label next to each field
QFormLayout form(&dialog);

// Add some text above the fields
form.addRow(new QLabel("The question ?"));

// Add the lineEdits with their respective labels
QList<QLineEdit *> fields;
for(int i = 0; i < 4; ++i) {
    QLineEdit *lineEdit = new QLineEdit(&dialog);
    QString label = QString("Value %1").arg(i + 1);
    form.addRow(label, lineEdit);

    fields << lineEdit;
}

// Add some standard buttons (Cancel/Ok) at the bottom of the dialog
QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                           Qt::Horizontal, &dialog);
form.addRow(&buttonBox);
QObject::connect(&buttonBox, SIGNAL(accepted()), &dialog, SLOT(accept()));
QObject::connect(&buttonBox, SIGNAL(rejected()), &dialog, SLOT(reject()));

// Show the dialog as modal
if (dialog.exec() == QDialog::Accepted) {
    // If the user didn't dismiss the dialog, do something with the fields
    foreach(QLineEdit * lineEdit, fields) {
        qDebug() << lineEdit->text();
    }
}