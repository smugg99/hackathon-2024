// components/GlitchedText.js
import React from 'react';
import styles from './glitched-text.module.css';

const GlitchedText = ({ text }: { text: string }) => {
  return (
    <div className={styles.glitched}>
      {text.split('').map((char, index) => (
        <span key={index} className={styles.char}>
          {char}
        </span>
      ))}
    </div>
  );
};

export default GlitchedText;
