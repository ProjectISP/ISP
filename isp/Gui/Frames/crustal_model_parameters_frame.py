from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiCrustalModelParametersFrame


class CrustalModelParametersFrame(pw.QDialog, UiCrustalModelParametersFrame):
    def __init__(self):
        super(CrustalModelParametersFrame, self).__init__()
        self.setupUi(self)

        self._orderWidgetsList = []
        self.addBtn.clicked.connect(self.on_add_action_pushed)

    def on_add_action_pushed(self):
        PB_del = pw.QPushButton("-")
        layoutPB = pw.QHBoxLayout()
        layoutPB.addWidget(PB_del)
        order_widget = pw.QWidget()
        layoutPB.setContentsMargins(0, 0, 0, 0)
        order_widget.setLayout(layoutPB)
        PB_del.clicked.connect(lambda parent=order_widget: self.removeRow(order_widget))

        self._orderWidgetsList.append(order_widget)
        param_widget = pw.QWidget()
        param_layout = pw.QHBoxLayout()
        param_layout.setContentsMargins(0,0,0,0)
        param_widget.setLayout(param_layout)

        self.parameters_table.setRowCount(self.parameters_table.rowCount() + 1)
        self.parameters_table.setCellWidget(self.parameters_table.rowCount() - 1, 0, order_widget)

        for i in range(1,self.parameters_table.columnCount()):
            item = pw.QTableWidgetItem()
            item.setData(0, 0.0)
            self.parameters_table.setItem(self.parameters_table.rowCount() - 1, i, item)


    def removeRow(self, order_widget):
        current_row = self._orderWidgetsList.index(order_widget)
        if current_row == ValueError:
            return

        self.parameters_table.removeRow(current_row)
        self._orderWidgetsList.pop(current_row)


    def getParameters(self):
        parameters = []
        for i in range(self.parameters_table.rowCount()) :
            row = []
            for j in range(1, self.parameters_table.columnCount()):
                row.append(str(self.parameters_table.item(i, j).data(0)))
            parameters.append(row)
        return parameters

    def getParametersWithFormat(self) :
        parameters = 'Crustal model                Rigo (my rho)\nnumber of layers\n'
        parameters += (str(self.parameters_table.rowCount()) + '\n')
        parameters += 'Parameters of the layers\ndepth of layer top(km)   Vp(km/s)    Vs(km/s)    Rho(g/cm**3)    Qp     Qs\n'
        
        for i in range(self.parameters_table.rowCount()) :
            for j in range(1, self.parameters_table.columnCount()):
                parameters += (str(self.parameters_table.item(i, j).data(0)) + ' ')
            parameters += '\n'

        parameters += '*******************************************************************\n'
        return parameters