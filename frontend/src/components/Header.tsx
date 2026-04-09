import { useTranslation } from 'react-i18next';
import LanguageToggle from './LanguageToggle';
import './Header.css';

function Header() {
  const { t } = useTranslation();

  return (
    <header className="header">
      <div className="header-content">
        <h1 className="title">{t('header.title')}</h1>
        <p className="subtitle">{t('header.subtitle')}</p>
      </div>
      <LanguageToggle />
    </header>
  );
}

export default Header;
