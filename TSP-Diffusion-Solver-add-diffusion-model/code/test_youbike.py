import unittest

from youbike import parse_candidates


class YouBikeTests(unittest.TestCase):
    def test_filters_and_preserves_station_identity(self):
        station = {"sno": "500601001", "sna": "YouBike station", "scityen": "Taichung City",
                   "act": 1, "lng": "120.68", "lat": "24.14", "sbi": "0"}
        rows = [station, station.copy(), dict(station, sno="2", act=0),
                dict(station, sno="3", scityen="Taipei City"),
                dict(station, sno="4", lat="nan"), dict(station, sno="5", lng="bad")]
        result = parse_candidates({"retCode": 1, "retVal": rows})
        self.assertEqual(result, [{"id": "youbike/500601001", "name": "YouBike station",
                                   "original_coords": [120.68, 24.14]}])

    def test_rejects_failed_or_empty_feed(self):
        for document in ({"retCode": 0}, {"retCode": 1, "retVal": []}):
            with self.assertRaises(ValueError):
                parse_candidates(document)


if __name__ == "__main__":
    unittest.main()
