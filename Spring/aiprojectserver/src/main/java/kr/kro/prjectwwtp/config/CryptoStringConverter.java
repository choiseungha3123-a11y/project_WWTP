package kr.kro.prjectwwtp.config;

import java.security.Key;
import java.security.SecureRandom;
import java.util.Base64;

import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import lombok.RequiredArgsConstructor;

@Service
@Converter
@RequiredArgsConstructor
public class CryptoStringConverter implements AttributeConverter<String, String> {
	@Value("${db.cryp.key}")
	private String crypKey;
	
	private String encode = "UTF-8";
	private static final String ALGORITHM = "AES/CBC/PKCS5Padding";

	@Override
	public String convertToDatabaseColumn(String attribute) {
		// TODO Auto-generated method stub
		if(attribute == null) return null;
		try {
			return encAES(attribute);
		}catch(Exception e) {
			return null;
		}
	}

	@Override
	public String convertToEntityAttribute(String dbData) {
		// TODO Auto-generated method stub
		if(dbData == null) return null;
		try {
			return decAES(dbData);
		}catch(Exception e) {
			return null;
		}
	}
	
	
	private Key getAESKey() throws Exception {
        byte[] bytes = new byte[32]; // AES-256
        byte[] keyBytes = crypKey.getBytes(encode);
        System.arraycopy(keyBytes, 0, bytes, 0, Math.min(keyBytes.length, bytes.length));
        return new SecretKeySpec(bytes, "AES");
    }
	
	// 암호화
	private String encAES(String str) throws Exception {
		// 랜덤 IV 생성
        byte[] iv = new byte[16];
        new SecureRandom().nextBytes(iv);
        IvParameterSpec ivSpec = new IvParameterSpec(iv);

        Cipher cipher = Cipher.getInstance(ALGORITHM);
        cipher.init(Cipher.ENCRYPT_MODE, getAESKey(), ivSpec);

        byte[] encrypted = cipher.doFinal(str.getBytes(encode));

        // IV + 암호문 합쳐서 Base64 저장
        byte[] combined = new byte[iv.length + encrypted.length];
        System.arraycopy(iv, 0, combined, 0, iv.length);
        System.arraycopy(encrypted, 0, combined, iv.length, encrypted.length);

        return Base64.getEncoder().encodeToString(combined);
	}
	
	// 복호화
	public String decAES(String str) throws Exception {
		byte[] combined = Base64.getDecoder().decode(str);

        // IV 추출 (앞 16바이트)
        byte[] iv = new byte[16];
        byte[] encrypted = new byte[combined.length - 16];
        System.arraycopy(combined, 0, iv, 0, 16);
        System.arraycopy(combined, 16, encrypted, 0, encrypted.length);

        IvParameterSpec ivSpec = new IvParameterSpec(iv);

        Cipher cipher = Cipher.getInstance(ALGORITHM);
        cipher.init(Cipher.DECRYPT_MODE, getAESKey(), ivSpec);

        return new String(cipher.doFinal(encrypted), encode);
	}
	

}
