CATEGORY_ID_LIST = [
    '0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007',
    '0008', '0100', '0101', '0102', '0103', '0104', '0200', '0201',
    '0202', '0203', '0204', '0205', '0206', '0207', '0208', '0300',
    '0301', '0302', '0303', '0304', '0305', '0400', '0401', '0402',
    '0403', '0404', '0405', '0500', '0501', '0502', '0600', '0601',
    '0602', '0700', '0701', '0702', '0703', '0704', '0705', '0800',
    '0801', '0802', '0803', '0804', '0805', '0900', '0901', '0902',
    '0903', '0904', '0905', '0906', '0907', '1000', '1001', '1002',
    '1003', '1100', '1101', '1102', '1103', '1104', '1105', '1200',
    '1201', '1202', '1203', '1204', '1205', '1300', '1301', '1302',
    '1303', '1304', '1305', '1306', '1307', '1308', '1309', '1310',
    '1311', '1400', '1401', '1402', '1403', '1500', '1501', '1502',
    '1503', '1504', '1505', '1506', '1507', '1508', '1509', '1600',
    '1601', '1602', '1603', '1604', '1605', '1606', '1607', '1608',
    '1609', '1610', '1700', '1701', '1702', '1703', '1704', '1705',
    '1706', '1707', '1708', '1800', '1801', '1802', '1803', '1804',
    '1805', '1806', '1900', '1901', '1902', '1903', '1904', '1905',
    '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',
    '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015',
    '2100', '2101', '2102', '2103', '2104', '2105', '2106', '2107',
    '2108', '2109', '2110', '2111', '2112', '2113', '2114', '2115',
    '2116', '2117', '2118', '2119', '2120', '2121', '2122', '2123',
    '2200', '2201', '2202', '2203', '2204', '2205', '2206', '2207',
    '2208', '2209', '2210', '2211', '2212', '2213', '2214', '2215',
    '2216', '2217', '2218', '2219', '2220', '2221', '2222', '2223'
]

cate_l2_num = [9, 5, 9, 6, 6, 3, 3, 6, 6, 8, 4, 6, 6, 12, 4, 10, 11, 9, 7, 6, 16, 24, 24]

CATEGORY_ID_TO_LV2ID = {k: v for v, k in enumerate(CATEGORY_ID_LIST)}
LV2ID_TO_CATEGORY_ID = {v: k for v, k in enumerate(CATEGORY_ID_LIST)}


def category_id_to_lv1id(category_id: str) -> int:
    """ Convert string category_id to level-1 class id. """
    return int(category_id[0:2])


def category_id_to_lv2id(category_id: str) -> int:
    """ Convert string category_id to level-2 class id. """
    return CATEGORY_ID_TO_LV2ID[category_id]


def lv2id_to_category_id(lv2id: int) -> str:
    """ Convert level-2 class id to string category_id. """
    return LV2ID_TO_CATEGORY_ID[lv2id]


def lv2id_to_lv1id(lv2id: int) -> int:
    """ Convert level-2 class id to level-1 class id. """
    category_id = lv2id_to_category_id(lv2id)
    return category_id_to_lv1id(category_id)
