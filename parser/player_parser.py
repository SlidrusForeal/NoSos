import asyncio
import aiohttp
from bs4 import BeautifulSoup
import logging
import re
from datetime import datetime
from telegram.helpers import escape_markdown

class PlayerParser:
    BASE_URL = "https://serverchichi.online/player/"
    _cache = {}

    @staticmethod
    async def fetch_player_page(player_name: str) -> str:
        url = f"{PlayerParser.BASE_URL}{player_name}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logging.error(f"Ошибка запроса {url}: {response.status}")
                    return ""
                return await response.text()

    @classmethod
    async def parse_player_profile(cls, player_name: str) -> dict:
        cache_key = player_name.lower()
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        html_content = await cls.fetch_player_page(player_name)
        if not html_content:
            return {}
        soup = BeautifulSoup(html_content, 'html.parser')
        player_data = {}
        socials_section = soup.find('div', class_='socials')
        player_data['socials'] = []
        if socials_section:
            for a in socials_section.find_all('a'):
                text = a.get_text(strip=True)
                href = a.get('href', '')
                if text and href:
                    player_data['socials'].append((text, href))
        roles_section = soup.find('div', class_='roles')
        player_data['roles'] = [role.get_text(strip=True) for role in roles_section.find_all('span')] if roles_section else []
        stats_section = soup.find('div', class_='stats')
        player_data['stats'] = [stat.get_text(strip=True) for stat in stats_section.find_all('p')] if stats_section else []
        rp_container = soup.find('div', class_='rp-container')
        if rp_container:
            player_data['rp_cards'] = [
                {'h3': card.find('h3').get_text(strip=True) if card.find('h3') else 'Без названия',
                 'p': card.find('p').get_text(strip=True) if card.find('p') else 'Нет данных'}
                for card in rp_container.find_all('div', class_='rp-card')
            ]
        else:
            player_data['rp_cards'] = []
        premium_section = soup.find('div', class_='player-plus-content')
        if premium_section:
            premium_text = premium_section.get_text(strip=True)
            match = re.search(r'СЧ\+\s*(\d+)\s*Уровня', premium_text)
            if match:
                player_data['player_plus'] = f"СЧ+ {match.group(1)} Уровня"
            else:
                player_data['player_plus'] = "СЧ+ пока не куплен"
        else:
            player_data['player_plus'] = "СЧ+ пока не куплен"
        cls._cache[cache_key] = player_data
        return player_data

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
