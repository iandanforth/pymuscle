import time
import asyncio


async def bar():
    print('Waiting 3 seconds ...')
    await asyncio.sleep(3)
    print('Here!')


async def main():
    counter = 0
    while True:
        if (counter % 10) != 0:
            time.sleep(0.5)
            asyncio.ensure_future(bar())
        else:
            await asyncio.sleep(0.5)
        print(counter)
        counter += 1


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(main())
    loop.run_forever()
