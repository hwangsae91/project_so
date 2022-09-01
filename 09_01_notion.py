def total_silver(treasure_box):
    total_coins = 0

    for k, v in treasure_box.items():
        total_item_coins = v['coin'] * v['pcs']
        total_coins += total_item_coins
        print(f"{k} : {v['coin']}coins/pcs * {v['pcs']}pcs = {total_item_coins} coins")

    print(f"total_coins : {total_coins}")
  
total_silver(treasure_box)
