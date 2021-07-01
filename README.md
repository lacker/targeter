# targeter
A tool to optimize target selection for telescopes.

To run this, first get a `dump.rdb` file from someone operating this live. I needed at least Redis 5 for modern dump file compatibility.

Then run redis against it. You need to copy in the dump file while Redis is stopped. Something like:

```
sudo systemctl stop redis.service
sudo cp ~/Downloads/dump.rdb /var/lib/redis/
sudo systemctl start redis.service
```
