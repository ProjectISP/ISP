from isp.Gui.Frames import BaseFrame, UiEarthquakeAnalysisFrame, Pagination
from isp.earthquakeAnalisysis import EarthquakeLocation


class EarthquakeAnalysisFrame(BaseFrame,UiEarthquakeAnalysisFrame):

    def __init__(self, ):
        super(EarthquakeAnalysisFrame, self).__init__()
        self.setupUi(self)

        self.test1Button.clicked.connect(self.onClick_test1)
        self.pagination = Pagination(self.pagination_widget, 1)
        self.pagination.set_total_items(10)
        self.pagination.bind_onPage_changed(self.onChange_page)
        self.pagination.bind_onItemPerPageChange_callback(self.onChange_itens_per_page)


    def onClick_test1(self):
        quake_location = EarthquakeLocation()
        quake_location.locate_earthquake(1)

    def onChange_page(self, page):
        print("Page num: ", page)
        print(self.pagination.items_per_page)

    def onChange_itens_per_page(self, v):
        print("Items per page: ", v)
        print(self.pagination.items_per_page)




